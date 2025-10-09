#!/usr/bin/env python3
"""
Newsletter Agent for testing the complete workflow end-to-end.

This agent implements all 9 workflow steps defined in the WorkflowStatus object.
Each step updates the workflow status and serializes the state.

"""

from agents import (Agent, Runner, RunContextWrapper, FunctionTool,
                    SQLiteSession, set_default_openai_client)
from do_rating import fn_rate_articles, bradley_terry
from do_cluster import do_clustering
from do_dedupe import process_dataframe_with_filtering
from llm import LLMagent, LangfuseClient
from utilities import send_gmail
from scrape import scrape_urls_concurrent, normalize_html
from fetch import Fetcher
from db import Url, Site
from log_handler import SQLiteLogHandler
from newsletter_state import NewsletterAgentState
from config import CANONICAL_TOPICS, DOWNLOAD_DIR, PAGES_DIR, TEXT_DIR, NEWSAGENTDB, LOGDB, DEFAULT_CONCURRENCY
from openai import AsyncOpenAI
import asyncio
import time
import logging
import os
import json
import dotenv
# import random
from datetime import datetime
from pathlib import Path
# from collections import Counter
import sqlite3

# from httpx import head
import tldextract

import shutil
# import pickle
# import numpy as np
import pandas as pd
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# import hdbscan

from IPython.display import display  # HTML, Image, Markdown,

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
# from email.utils import parsedate_to_datetime
# from urllib.parse import urlparse
from bs4 import BeautifulSoup


# Pydantic models for section drafting
class SectionStoryLink(BaseModel):
    url: str = Field(description="URL of the article")
    site_name: str = Field(description="Name of the website/source")

    def __str__(self):
        return f"[{self.site_name}]({self.url})"


class Story(BaseModel):
    headline: str = Field(description="Summary of the story")
    links: List[SectionStoryLink] = Field(
        description="List of links related to this story")
    prune: bool = Field(description="Whether to prune/exclude this story")

    def __str__(self):
        return f"- {self.headline} - " + " ".join([str(s) for s in self.links])


class Section(BaseModel):
    section_title: str = Field(description="Title of the newsletter section")
    headlines: List[Story] = Field(
        description="List of stories in this section")

    def __str__(self):
        return f"## {self.section_title}\n\n" + "\n".join(
            [str(h) for h in self.headlines if not h.prune]
        )


# Pydantic models for structured output


class ArticleSummary(BaseModel):
    """Model for AI-generated article summaries with exactly 3 bullet points"""
    id: int = Field(description="The article id")
    summary: str = Field(
        description="Bullet-point summary of the article"
    )


class ArticleSummaryList(BaseModel):
    """List of AIClassification for batch processing"""
    results_list: list[ArticleSummary] = Field(
        description="List of summary results")

# output class for classifying headlines


class AIClassification(BaseModel):
    """A single headline classification result"""
    id: int = Field("The news item id")
    input_str: str = Field(description="The original headline title")
    output: bool = Field(
        description="Whether the headline title is AI-related")


class AIClassificationList(BaseModel):
    """List of AIClassification for batch processing"""
    results_list: list[AIClassification] = Field(
        description="List of classification results")

# Topic extraction models


class TopicExtraction(BaseModel):
    """Topic extraction result for a single article"""
    id: int = Field(description="The article id")
    topics_list: List[str] = Field(
        description="List of relevant topics discussed in the article")


class TopicExtractionList(BaseModel):
    """List of TopicExtraction for batch processing"""
    results_list: list[TopicExtraction] = Field(
        description="List of topic extraction results")

# Canonical topic classification models


class CanonicalTopicClassification(BaseModel):
    """Single article classification result for a canonical topic"""
    id: int = Field(description="The article id")
    relevant: bool = Field(
        description="Whether the summary is relevant to the canonical topic")


class CanonicalTopicClassificationList(BaseModel):
    """List of classification results for batch processing"""
    results_list: list[CanonicalTopicClassification] = Field(
        description="List of classification results")

# Site name generation models


class SiteNameGeneration(BaseModel):
    """Single domain to site name mapping result"""
    id: int = Field(description="The site id")
    domain: str = Field(description="The domain name")
    site_name: str = Field(description="Canonical site name for the domain")


class SiteNameGenerationList(BaseModel):
    """List of SiteNameGeneration for batch processing"""
    results_list: list[SiteNameGeneration] = Field(
        description="List of site name generation results")

# for categorizing articles


class TopicHeadline(BaseModel):
    """Topic headline of a group of stories for structured output"""
    topic_title: str = Field(description="The title for the headline group")


class TopicCategoryList(BaseModel):
    """List of topics for structured output filtering"""
    items: List[str] = Field(description="List of topics")

# for deduping articles


class DupeRecord(BaseModel):
    """Dupe record for structured output filtering"""
    id: int = Field(description="The article id")
    dupe_id: int = Field(description="The dupe article id")


class DupeRecordList(BaseModel):
    """List of DupeRecord for structured output filtering"""
    results_list: list[DupeRecord] = Field(
        description="List of dupe records")


class DistilledStory(BaseModel):
    """DistilledStory class for structured output distillation into a single sentence """
    id: int = Field(description="The article id")
    short_summary: str = Field(description="The distilled short summary")


class DistilledStoryList(BaseModel):
    """List of DistilledStory for batch processing"""
    results_list: list[DistilledStory] = Field(
        description="List of distilled stories")


def setup_logging(session_id: str = "default", db_path: str = LOGDB) -> logging.Logger:
    """Set up logging to console and SQLite database."""

    # Create logger
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(f"NewsletterAgent.{session_id}")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # SQLite handler
    sqlite_handler = SQLiteLogHandler(db_path)
    sqlite_handler.setLevel(logging.INFO)
    sqlite_formatter = logging.Formatter('%(message)s')
    sqlite_handler.setFormatter(sqlite_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(sqlite_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


async def mycreate_task(delay_index: int, coro, delay_seconds: float = 0.1):
    """
    Create a task with a staggered delay based on index.
    was seeing some timeout errors, so added this to stagger the tasks
    not currently using it, wraps tasks in a short delay to stagger them

    Args:
        delay_index: Index used to calculate delay (delay_index * delay_seconds)
        coro: Coroutine to execute after delay
        delay_seconds: Base delay in seconds (default: 0.1)

    Returns:
        Result of the coroutine after the calculated delay
    """
    delay = delay_index * delay_seconds
    if delay > 0:
        await asyncio.sleep(delay)
    return await coro

# tools


class WorkflowStatusTool:
    """Tool to check current workflow status"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    async def _check_workflow_status(self, ctx, args: str) -> str:
        """Get current workflow status report based on persistent state"""
        if self.logger:
            self.logger.info("Starting check_workflow_status")

        try:
            # Access the persistent state
            state: NewsletterAgentState = ctx.context

            # Use the unified status reporting system
            result = state.get_workflow_status_report(
                "WORKFLOW STATUS (FROM PERSISTENT STATE)")

            # Add data summary if we have articles
            if state.headline_data:

                ai_related = sum(
                    1 for a in state.headline_data if a.get('isAI') is True)
                result += "\n\nData Summary:\n"
                result += f"  Total articles: {len(state.headline_data)}\n"
                result += f"  AI-related: {ai_related}\n"
                result += f"  Clusters: {len(state.clusters)}\n"
                result += f"  Sections: {len(state.newsletter_section_text)}"

            # Add intervention guidance if workflow is in error state
            if state.has_errors():
                failed_steps = state.get_failed_steps()
                result += "\n\n⚠️  INTERVENTION REQUIRED:\n"
                result += f"  Failed steps: {', '.join(failed_steps)}\n"
                if state.workflow_status_message:
                    result += f"  Instructions: {state.workflow_status_message}"

            if self.logger:
                self.logger.info("Completed check_workflow_status")

            # Serialize state after checking workflow status
            state.serialize_to_db("check_workflow_status")
            return result

        except Exception as e:
            if self.logger:
                self.logger.error(f"check_workflow_status failed: {str(e)}")
            raise

    def create_tool(self) -> FunctionTool:
        """Create a FunctionTool instance following OpenAI Agents SDK conventions"""
        return FunctionTool(
            name="check_workflow_status",
            description="Check the current status of the newsletter workflow and see which steps are completed, in progress, or pending",
            params_json_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            on_invoke_tool=self._check_workflow_status
        )


class StateInspectionTool:
    """Tool to inspect detailed persistent state data"""

    def __init__(self, verbose: bool = False, logger: logging.Logger = None):
        self.verbose = verbose
        self.logger = logger

    async def _inspect_state(self, ctx, args: str) -> str:
        """Inspect detailed state data for debugging and monitoring"""
        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Create detailed state report using unified status system
        report_lines = [
            "DETAILED STATE INSPECTION",
            "=" * 50,
            f"Current Step: {state.get_current_step()}",
            f"Workflow Complete: {state.all_complete()}",
            f"Progress: {state.get_progress_percentage():.1f}%",
            f"Workflow Status: {state.workflow_status.value}",
        ]

        if state.workflow_status_message:
            report_lines.append(
                f"Status Message: {state.workflow_status_message}")

        report_lines.extend([
            f"Sources File: {state.sources_file}",
            "",
            "HEADLINE DATA:",
            f"  Total articles: {len(state.headline_data)}",
        ])

        if state.headline_data:
            ai_related = sum(
                1 for a in state.headline_data if a.get('isAI') is True)
            with_content = sum(
                1 for a in state.headline_data if a.get('content'))
            with_ratings = sum(
                1 for a in state.headline_data if a.get('quality_rating'))
            with_clusters = sum(
                1 for a in state.headline_data if a.get('cluster_topic'))

            report_lines.extend([
                f"  AI-related: {ai_related}",
                f"  With content: {with_content}",
                f"  With ratings: {with_ratings}",
                f"  With clusters: {with_clusters}",
                f"  Sources: {len(set(a.get('source', 'Unknown') for a in state.headline_data))}",
            ])

        report_lines.extend([
            "",
            "PROCESSING RESULTS:",
            f"  Topic clusters: {len(state.clusters)} topics",
            f"  Newsletter sections: {len(state.newsletter_section_text)} sections",
            f"  Final newsletter: {'Generated' if state.final_newsletter else 'Not created'}",
        ])

        if state.clusters:
            report_lines.extend([
                "",
                "TOPIC CLUSTERS:",
            ])
            for topic, urls in state.clusters.items():
                report_lines.append(f"  {topic}: {len(urls)} articles")

        if state.newsletter_section_text:
            report_lines.extend([
                "",
                "NEWSLETTER SECTIONS:",
            ])
            for section_name, section_content in state.newsletter_section_text.items():
                word_count = len(section_content.split())
                report_lines.append(
                    f"  {section_name}: {word_count} words")

        if state.final_newsletter:
            newsletter_words = len(state.final_newsletter.split())
            report_lines.extend([
                "",
                "FINAL NEWSLETTER:",
                f"  Length: {newsletter_words} words",
                f"  Preview: {state.final_newsletter[:200]}..." if len(
                    state.final_newsletter) > 200 else f"  Content: {state.final_newsletter}",
            ])

        # Serialize state after inspecting state
        state.serialize_to_db("inspect_state")
        return "\n".join(report_lines)

    def create_tool(self) -> FunctionTool:
        """Create a FunctionTool instance following OpenAI Agents SDK conventions"""
        return FunctionTool(
            name="inspect_state",
            description="Inspect detailed persistent state data including article counts, processing results, and content status. Useful for debugging and monitoring workflow progress.",
            params_json_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            on_invoke_tool=self._inspect_state
        )


class GatherUrlsTool:
    """Tool for Step 1: Gather URLs from news sources defined in sources.yaml using Fetcher
    - store headlines in persistent headline_data
    """

    def __init__(self, verbose: bool = False, logger: logging.Logger = None):
        self.verbose = verbose
        self.logger = logger

    def clean_download_dir(self, download_dir: str):
        """Clean up download directory"""
        if os.path.exists(download_dir):
            if self.logger:
                self.logger.info(f"Cleaning {download_dir}: ")

            # Remove all files and subdirectories in download_dir
            try:
                for item in os.listdir(download_dir):
                    item_path = os.path.join(download_dir, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                if self.logger:
                    self.logger.info(f"Successfully cleaned {download_dir}")
            except Exception as cleanup_error:
                if self.logger:
                    self.logger.warning(
                        f"Failed to clean {download_dir}: {cleanup_error}")

    async def _fetch_urls(self, ctx, args: str) -> str:
        """Execute Step 1: Gather URLs using persistent state"""
        if self.logger:
            self.logger.info("Starting Step 1: Gather URLs")

        step_name = "step_01_fetch_urls"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.is_step_complete("step_01_fetch_urls"):
            total_articles = len(state.headline_data)
            if self.logger:
                self.logger.info(
                    f"Step 1 already completed with {total_articles} articles")
            return f"Step 1 already completed! Found {total_articles} articles in persistent state."

        try:
            # Start the step using unified status system
            state.start_step(step_name)
            print(f"▶ Starting Step 1: {step_name}")

            # Clean up download directory if do_download flag is set
            if state.do_download:
                self.clean_download_dir(DOWNLOAD_DIR)
                self.clean_download_dir(TEXT_DIR)
                self.clean_download_dir(PAGES_DIR)

            # Use RSS fetching from sources.yaml
            # sources_config = None
            async with Fetcher() as f:
                sources_results = await f.fetch_all(do_download=state.do_download)
                # Capture sources configuration for later reference (e.g., failed source URLs)
                # sources_config = dict(getattr(f, 'sources', {}) or {})

            # Process results and store in persistent state
            successful_sources = []
            failed_sources = []
            all_articles = []

            for result in sources_results:
                if result['status'] == 'success' and result['results']:
                    # Add source info to each article
                    successful_sources.append(result['source'])
                    all_articles.extend(result['results'])
                else:
                    failed_sources.append(result['source'])

            # Check if we need user intervention (something failed)
            # total_sources = len(successful_sources) + len(failed_sources)
            # success_rate = len(successful_sources) / \
            #     total_sources if total_sources > 0 else 0

            # Define intervention scenarios (currently unused but kept for clarity)
            # requires_intervention = success_rate < 1.0

            if failed_sources:
                # Set error state with intervention message
                intervention_message = f"Partial failure detected. {len(failed_sources)} sources failed: {', '.join(failed_sources)}. "

                # show urls for failed sources from state.sources
                failed_source_urls = []
                if state.sources:
                    for src in failed_sources:
                        cfg = state.sources.get(src) or {}
                        url = cfg.get('rss') or cfg.get('url') or ''
                        if url:
                            failed_source_urls.append(f"{src}: {url}")
                if failed_source_urls:
                    intervention_message += f"Failed source URLs: {', '.join(failed_source_urls)}. "
                intervention_message += "Download HTML files manually to download/sources/ directory, clear error and restart with do_download=False."

                state.error_step(step_name, intervention_message)

                if self.verbose:
                    print(
                        f"⚠️  Intervention required: {len(successful_sources)} successful, {len(failed_sources)} failed")
                    print(f"Failed sources: {', '.join(failed_sources)}")
                    if failed_source_urls:
                        print(
                            f"Failed source URLs: {', '.join(failed_source_urls)}")

                return f"⚠️  Intervention Required! Successfully fetched from {len(successful_sources)} sources but {len(failed_sources)} sources failed.\n\n{intervention_message}"

            # Successfully fetched URLs, store results in persistent state
            headline_df = pd.DataFrame(all_articles)
            headline_df = headline_df.drop_duplicates(
                subset=['url'], keep='first')
            headline_df['id'] = range(len(headline_df))

            display(headline_df[["source", "url"]].groupby("source")
                    .count()
                    .reset_index()
                    .rename({'url': 'count'}))
            state.headline_data = headline_df.to_dict('records')

            # Complete the step using unified status system
            state.complete_step(step_name)

            if self.verbose:
                print(
                    f"✅ Completed Step 1: Gathered {len(all_articles)} URLs from {len(successful_sources)} RSS sources")
                if failed_sources:
                    print(f"⚠️  Failed sources: {', '.join(failed_sources)}")

            status_msg = f"✅ Step 1 {step_name} completed successfully! Gathered {len(all_articles)} articles from {len(successful_sources)} sources (RSS only)."
            if failed_sources:
                status_msg += f" {len(failed_sources)} sources failed or not implemented."

            status_msg += f"\n\n📊 Articles stored in persistent state: {len(state.headline_data)}"
            headline_df = pd.DataFrame(state.headline_data)

            display(headline_df)

            if self.logger:
                self.logger.info(
                    f"Completed Step 1: Gathered {len(all_articles)} articles")

            # Serialize state after completing step
            state.serialize_to_db(step_name)
            return status_msg

        except Exception as e:
            error_msg = f"Failed to fetch URLs: {str(e)}"
            if self.logger:
                self.logger.error(f"Step 1 failed: {str(e)}")

            # Set error state using unified status system
            state.error_step(step_name, error_msg)
            return f"❌ Step 1 failed: {error_msg}"

    def create_tool(self) -> FunctionTool:
        """Create a FunctionTool instance following OpenAI Agents SDK conventions"""
        return FunctionTool(
            name="fetch_urls",
            description="Execute Step 1: Gather URLs and headlines from various news sources. Only use this tool if Step 1 is not already completed.",
            params_json_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            on_invoke_tool=self._fetch_urls
        )


class FilterUrlsTool:
    """Tool for Step 2: Filter URLs to AI-related content
    only download articles that are AI-related that are not already downloaded
    - get all source, url, title from headline_data
    - check if previously downloaded same url
    - check if if previously downloaded same source, title (eg google news may rotate url)
    - if not downloaded,
    - store in downloaded_articles db
    - check all titles if headline is AI-related using prompt
    - merge this step with downloaded_articles step
    """

    def __init__(self, verbose: bool = False, logger: logging.Logger = None):
        self.verbose = verbose
        self.logger = logger

    async def _filter_urls(self, ctx, args: str) -> str:
        """Execute Step 2: Filter URLs using persistent state"""
        if self.logger:
            self.logger.info("Starting Step 2: Filter URLs")

        step_name = "step_02_filter_urls"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context
        # Check if step already completed via persistent state
        if state.is_step_complete(step_name):
            ai_related_count = sum(
                1 for article in state.headline_data if article.get('isAI') is True)
            total_count = len(state.headline_data)
            return f"Step 2 already completed! Filtered {total_count} articles, {ai_related_count} identified as AI-related."

        # Check if step 1 is completed and no errors
        if not state.is_step_complete("step_01_fetch_urls") or not state.headline_data:
            return f"❌ Cannot execute Step 2: Step 1 (Gather URLs) must be completed first. Current status: {state.get_current_step()}"

        # Check if workflow is blocked by errors
        if state.has_errors():
            return f"❌ Cannot execute Step 2: Workflow is blocked by errors. {state.workflow_status_message}"

        try:
            # Start step
            state.start_step(step_name)
            print(f"▶ Starting Step 2: {step_name}")

            # Read headlines from persistent state
            total_articles = len(state.headline_data)

            if self.logger:
                self.logger.info(f"🔍 Filtering {total_articles} headlines...")
                if state.process_since:
                    self.logger.info(
                        f"🔄 Checking for duplicates seen before {state.process_since.isoformat()}")
                else:
                    self.logger.info(
                        "🔄 Checking for duplicates (all urls without date restrictions)")

            # Prepare headlines for batch classification
            headline_df = pd.DataFrame(state.headline_data)

            # Filter out URLs that have already been seen (URL deduplication)
            original_count = len(headline_df)

            if self.logger:
                self.logger.info(
                    f"🔍 Filtering {total_articles} articles for dupes.")

            try:
                with sqlite3.connect(state.db_path) as conn:
                    Url.create_table(conn)  # Ensure table exists
                    # Check each URL against the urls table
                    dupe_df = headline_df.copy()
                    dupe_df['is_new'] = True
                    for row in dupe_df.itertuples():
                        url_to_check = row.url
                        print("checking", url_to_check)
                        if not url_to_check:  # should never happen
                            continue
                        existing_url = Url.get_by_url_or_source_and_title(
                            conn, url_to_check, row.source, row.title)
                        if existing_url is None:  # not found
                            print("url not found - inserting new URL")
                            new_url = Url(
                                initial_url=url_to_check,
                                final_url='',
                                title=row.title,
                                source=row.source,
                                isAI=None,  # Will be set after AI classification
                                created_at=datetime.now()
                            )
                            new_url.insert(conn)
                            continue
                        if state.process_since:
                            if existing_url.created_at is not None and existing_url.created_at > state.process_since:
                                # URL exists but seen after process_since - treat as new
                                print("found url after cutoff")
                                continue
                            # found url and seen prior to cutoff - treat as duplicate
                            print("found before cutoff")
                            dupe_df.at[row.Index, 'is_new'] = False
                        else:
                            # No process_since set - treat any existing URLs as duplicates
                            print("found url, no cutoff set - treating as duplicate")
                            dupe_df.at[row.Index, 'is_new'] = False

                # Filter DataFrame to only new URLs
                headline_df = headline_df.loc[dupe_df['is_new']]
                duplicate_count = original_count - len(headline_df)

                if self.verbose:
                    if state.process_since is not None:
                        print(
                            f"🔄 Filtered out {duplicate_count} URLs seen before {state.process_since.isoformat()}, processing {len(headline_df)} new URLs")
                    else:
                        print(
                            f"🔄 Filtered out {duplicate_count} duplicate URLs, processing {len(headline_df)} new URLs")
                if self.logger:
                    if state.process_since is not None:
                        self.logger.info(
                            f"URL deduplication with process_since: {duplicate_count} URLs filtered (seen before {state.process_since.isoformat()}), {len(headline_df)} new URLs remain")
                    else:
                        self.logger.info(
                            f"URL deduplication: {duplicate_count} duplicates filtered, {len(headline_df)} new URLs remain")

                # If no new URLs remain, complete step early
                if headline_df.empty:
                    state.complete_step(step_name)
                    if state.process_since is not None:
                        return f"✅ Step 2 completed! All {original_count} URLs were seen before {state.process_since.isoformat()} - no new content to process."
                    else:
                        return f"✅ Step 2 completed! All {original_count} URLs were duplicates - no new content to process."

            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        f"URL deduplication failed: {e}. Proceeding without deduplication.")
                if self.verbose:
                    print(
                        f"⚠️ URL deduplication failed: {e}. Proceeding with all URLs.")

            if self.logger:
                self.logger.info(
                    f"🔍 Filtering {total_articles} headlines for AI relevance using LLM...")

            filter_system_prompt, filter_user_prompt, model = \
                LangfuseClient().get_prompt("newsagent/filter_urls")

            # Create LLM agent for AI classification
            classifier = LLMagent(
                system_prompt=filter_system_prompt,
                user_prompt=filter_user_prompt,
                output_type=AIClassificationList,
                model=model,
                verbose=self.verbose,
                logger=self.logger
            )

            headline_df['isAI'] = \
                await classifier.filter_dataframe(headline_df[["id", "title"]])

            # Update state with AI classification results
            state.headline_data = headline_df.to_dict('records')

            # Complete step using unified status system
            state.complete_step(step_name)

            ai_related_count = sum(
                1 for article in state.headline_data if article.get('isAI') is True)
            total_count = len(state.headline_data)
            duplicate_count = original_count - \
                total_count if 'original_count' in locals() else 0

            if self.verbose:
                if duplicate_count > 0:
                    print(
                        f"✅ Completed Step 2: {duplicate_count} duplicates removed, {ai_related_count} AI-related from {total_count} processed ({total_articles} original)")
                else:
                    print(
                        f"✅ Completed Step 2: Filtered to {ai_related_count} AI-related headlines from {total_articles} total")

            # Build status message with deduplication stats
            if duplicate_count > 0:
                status_msg = f"✅ Step 2 {step_name} completed successfully! Removed {duplicate_count} duplicate URLs, classified {total_count} new articles, found {ai_related_count} AI-related."
            else:
                status_msg = f"✅ Step 2 {step_name} completed successfully! Filtered {total_articles} headlines to {ai_related_count} AI-related articles."

            if self.logger:
                log_msg = f"Completed Step 2: {ai_related_count} AI-related articles"
                if duplicate_count > 0:
                    log_msg += f", {duplicate_count} duplicates removed"
                self.logger.info(log_msg)

            # Serialize state after completing step
            state.serialize_to_db(step_name)
            return status_msg

        except Exception as e:
            error_msg = f"Failed to filter URLs: {str(e)}"
            if self.logger:
                self.logger.error(f"Step 2 failed: {str(e)}")

            # Set error state using unified status system
            state.error_step(step_name, error_msg)
            return f"❌ Step 2 failed: {error_msg}"

    def create_tool(self) -> FunctionTool:
        """Create a FunctionTool instance following OpenAI Agents SDK conventions"""
        return FunctionTool(
            name="filter_urls",
            description="Execute Step 2: Filter URLs to AI-related content only. Requires Step 1 to be completed first.",
            params_json_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            on_invoke_tool=self._filter_urls
        )


class DownloadArticlesTool:
    """Tool for Step 3: Download article content
    - merge this step with filter_urls step
    - check if downloaded article already exists in download/articles directory
    - download url using playwright if not a domain that we gave up on like wsj due to blocks
    - store redirect url in headline_data
    - store publish date in headline_data
    - convert html to text with trafilatura, ignoring navigation, ads etc.
    - check if text is 90% similar to previous article via chromadb
    - if not similar, save text and update text_path to point to text, else "" and mark as dupe
    """

    def __init__(self, verbose: bool = False, logger: logging.Logger = None):
        self.verbose = verbose
        self.logger = logger

    async def _download_articles(self, ctx, args: str) -> str:
        """Execute Step 3: Download Articles using persistent state"""
        if self.logger:
            self.logger.info("Starting Step 3: Download Articles")

        step_name = "step_03_download_articles"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.is_step_complete(step_name):
            ai_related_count = sum(
                1 for article in state.headline_data if article.get('isAI') is True)
            total_count = len(state.headline_data)
            # todo: count downloaded articles with path configured
            # check if they exist in download directory
            return f"Step 3 already completed! Filtered {total_count} articles, {ai_related_count} identified as AI-related."

        # Check if step 2 is completed and no errors
        if not state.is_step_complete("step_02_filter_urls"):
            return f"❌ Cannot execute Step 3: Step 2 (Filter URLs) must be completed first. Current status: {step_name}"

        # Check if workflow is blocked by errors
        if state.has_errors():
            return f"❌ Cannot execute Step 3: Workflow is blocked by errors. {state.workflow_status_message}"

        try:
            # Update workflow status for UI tracking
            state.start_step(step_name)
            print(f"▶ Starting Step 3: {step_name}")

            headline_df = pd.DataFrame(state.headline_data)

            # Filter for AI-related articles
            ai_df = headline_df.loc[headline_df['isAI']].copy()

            if ai_df.empty:
                return "❌ No AI-related articles found to download. Please run step 2 first."

            # Prepare input for scrape_urls_concurrent: (index, url, title)
            scrape_inputs = []
            for idx, row in ai_df.iterrows():
                scrape_inputs.append((
                    row['id'],  # Use 'id' as the index for matching
                    row['url'],
                    row['title']
                ))

            if self.logger:
                self.logger.info(
                    f"Starting concurrent scraping of {len(scrape_inputs)} AI-related articles")

            # Use real scraping with scrape_urls_concurrent
            scrape_results = await scrape_urls_concurrent(
                scrape_inputs,
                concurrency=DEFAULT_CONCURRENCY,
                rate_limit_seconds=2.0,
                logger=self.logger
            )

            # Process scraping results and update DataFrame
            successful_downloads = 0
            total_length = 0

            # Create mapping from id to scrape results
            # Convert scrape results to DataFrame and merge with ai_df
            scrape_df = pd.DataFrame(scrape_results, columns=[
                                     'id', 'status', 'final_url', 'title', 'html_path', 'last_updated'])
            ai_df = ai_df.merge(scrape_df[[
                                'id', 'status', 'final_url', 'html_path', 'last_updated']], on='id', how='left')

            os.makedirs(TEXT_DIR, exist_ok=True)

            # Update headline_data with scraping results
            for row in ai_df.itertuples():
                if row.isAI is True and hasattr(row, 'html_path') and row.html_path:
                    try:
                        # Use normalize_html to extract clean text content
                        content = normalize_html(
                            row.html_path, logger=self.logger)

                        # Create text file path by replacing .html with .txt and moving to TEXT_DIR
                        html_file = Path(row.html_path)
                        text_filename = html_file.stem + '.txt'
                        text_path = os.path.join(TEXT_DIR, text_filename)

                        # Write content to text file
                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(content)

                        # Update DataFrame with text path and content length
                        ai_df.loc[row.Index, 'text_path'] = text_path
                        ai_df.loc[row.Index, 'content_length'] = len(content)
                        ai_df['content_length'] = ai_df['content_length'].fillna(
                            0).astype(int)
                        successful_downloads += 1
                        total_length += len(content)

                        if self.logger:
                            self.logger.debug(
                                f"Successfully extracted content to {text_path}: {len(content)} characters")
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(
                                f"Failed to extract content from {row.html_path}: {e}")
                        ai_df.loc[row.Index, 'text_path'] = ''
                        ai_df.loc[row.Index, 'content_length'] = 0
                else:
                    # No HTML file to process
                    ai_df.loc[row.Index, 'text_path'] = ''
                    ai_df.loc[row.Index, 'content_length'] = 0

            ai_df['text_path'] = ai_df['text_path'].fillna('')
            # Deduplicate by final_url, keeping the one with greater content_length
            ai_df['content_length'] = ai_df['content_length'].fillna(0)
            ai_df = ai_df.sort_values('content_length', ascending=False).drop_duplicates(
                subset=['final_url'], keep='first')

            # Extract domain name from final_url
            def extract_domain(url):
                """Extract root domain from URL (e.g., 'bbc.co.uk', 'archive.is')"""
                if not url or pd.isna(url):
                    return ''
                try:
                    extracted = tldextract.extract(str(url))
                    if extracted.domain:
                        domain = f"{extracted.domain}.{extracted.suffix}"
                    else:  # local hosts, ip addresses or other edge cases
                        domain = extracted.subdomain
                    return domain.lower()
                except Exception:
                    return ''

            ai_df['domain'] = ai_df['final_url'].fillna(
                ai_df['url']).apply(extract_domain)

            # Lookup site information from sites table
            unique_domains = ai_df['domain'].dropna().unique().tolist()
            domain_reputation_map = {}
            domain_site_name_map = {}

            if unique_domains:
                try:
                    with sqlite3.connect(NEWSAGENTDB) as conn:
                        Site.create_table(conn)  # Ensure sites table exists

                        # Use SQL IN clause for efficient batch lookup
                        placeholders = ','.join(['?'] * len(unique_domains))
                        cursor = conn.execute(f"""
                            SELECT domain_name, site_name, reputation
                            FROM sites
                            WHERE domain_name IN ({placeholders})
                        """, unique_domains)

                        # Create mapping dicts from query results
                        for domain_name, site_name, reputation in cursor.fetchall():
                            domain_site_name_map[domain_name] = site_name
                            domain_reputation_map[domain_name] = reputation

                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"Failed to lookup site information: {e}")

            # Map values to ai_df, filling missing domains with defaults
            ai_df['site_name'] = ai_df['domain'].map(
                domain_site_name_map).fillna('')
            ai_df['reputation'] = ai_df['domain'].map(
                domain_reputation_map).fillna(0)

            if self.logger:
                found_domains = len(domain_reputation_map)
                total_domains = len(unique_domains)
                self.logger.info(
                    f"Populated site information for {len(ai_df)} articles: {found_domains}/{total_domains} domains found in sites table")

            # Generate site names for domains with empty site_name using LLM
            domains_to_process = ai_df.loc[ai_df['site_name']
                                           == '']['domain'].drop_duplicates()

            if len(domains_to_process):
                if self.logger:
                    self.logger.info(
                        f"Generating site names for {len(domains_to_process)} domains using LLM")

                # Get prompt from Langfuse
                system_prompt, user_prompt, model = LangfuseClient().get_prompt("newsagent/sitename")

                # Create LLM agent for site name generation
                sitename_agent = LLMagent(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_type=SiteNameGenerationList,
                    model=model,
                    verbose=self.verbose,
                    logger=self.logger
                )

                # Create DataFrame for processing (just domain names)
                domains_df = pd.DataFrame(
                    {'domain': domains_to_process.tolist()})
                domains_df['id'] = range(len(domains_df))

                try:
                    # Generate site names using LLM with batch size of 25
                    domains_df['sitenames'] = await sitename_agent.filter_dataframe(
                        domains_df[['id', 'domain']],
                        value_field='site_name',
                        item_id_field='id',
                        chunk_size=25
                    )

                    # Create mapping of domain to generated site name
                    domain_to_sitename = dict(
                        zip(domains_df['domain'], domains_df['sitenames']))

                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"Failed to generate site names using LLM: {e}")
                    # Fallback: use domain as site_name
                    domain_to_sitename = {
                        domain: domain for domain in domains_to_process}

                # Update ai_df with generated site names (or fallback domain names)
                ai_df.loc[ai_df['site_name'] == '', 'site_name'] = ai_df.loc[ai_df['site_name'] == '', 'domain'].apply(
                    lambda x: domain_to_sitename.get(x, x))

                # Insert new site records into database
                try:
                    with sqlite3.connect(NEWSAGENTDB) as conn:
                        Site.create_table(conn)  # Ensure sites table exists

                        for domain, site_name in domain_to_sitename.items():
                            if site_name and site_name.strip():  # Only insert non-empty site names
                                site_record = Site(
                                    domain_name=domain,
                                    site_name=site_name.strip(),
                                    reputation=0.0  # Default reputation
                                )
                                try:
                                    site_record.upsert(conn)
                                except Exception as e:
                                    if self.logger:
                                        self.logger.warning(
                                            f"Failed to insert site record for {domain}: {e}")

                        if self.logger:
                            self.logger.info(
                                f"Inserted {len([s for s in domain_to_sitename.values() if s and s.strip()])} new site records")

                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"Failed to update sites table: {e}")

            # Update URLs table with final_url and isAI information
            urls_updated = 0
            try:
                with sqlite3.connect(NEWSAGENTDB) as conn:
                    Url.create_table(conn)  # Ensure table exists
                    for row in ai_df.itertuples():
                        try:
                            original_url = row.url
                            final_url = getattr(
                                row, 'final_url', original_url) or original_url
                            is_ai = getattr(row, 'isAI', False)

                            # Get existing URL record
                            existing_url = Url.get(conn, original_url)
                            if existing_url:
                                # Update with final URL and AI classification
                                existing_url.final_url = final_url
                                existing_url.isAI = is_ai
                                existing_url.update(conn)
                                urls_updated += 1

                        except Exception as e:
                            if self.logger:
                                self.logger.warning(
                                    f"Failed to update URL record for {original_url}: {e}")

                if self.logger and urls_updated > 0:
                    self.logger.info(
                        f"Updated {urls_updated} URL records with final URLs")

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to update URLs table: {e}")

            # dedupe by cosine similarity
            # several sources might syndicate same AP article, feedly and newsapi might show same article under different URL
            ai_df = await process_dataframe_with_filtering(ai_df)

            # Calculate stats
            download_success_rate = successful_downloads / \
                len(ai_df) if not ai_df.empty else 0
            avg_article_length = total_length / \
                successful_downloads if successful_downloads > 0 else 0

            # Store updated headline data in state
            state.headline_data = ai_df.to_dict('records')

            # Complete the step
            state.complete_step(step_name)

            if self.verbose:
                print(
                    f"✅ Completed Step 3: Downloaded {successful_downloads} AI-related articles")

            status_msg = f"✅ Step 3 {step_name} completed successfully! Downloaded {successful_downloads} AI-related articles with {download_success_rate:.0%} success rate."
            status_msg += f"\n📊 Average article length: {avg_article_length:.0f} characters"
            status_msg += "\n🔗 Content stored in persistent state."
            if self.logger:
                self.logger.info(
                    f"Completed Step 3: Downloaded {successful_downloads} articles")

            # Serialize state after completing step
            state.serialize_to_db(step_name)
            return status_msg

        except Exception as e:
            if self.logger:
                self.logger.error(f"Step 3 failed: {str(e)}")
            state.error_step(step_name, str(e))
            return f"❌ Step 3 failed: {str(e)}"

    def create_tool(self) -> FunctionTool:
        """Create a FunctionTool instance following OpenAI Agents SDK conventions"""
        return FunctionTool(
            name="download_articles",
            description="Execute Step 3: Download full article content from filtered URLs. Requires Step 2 to be completed first.",
            params_json_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            on_invoke_tool=self._download_articles
        )


class ExtractSummariesTool:
    """Tool for Step 4: Extract article summaries
    - if text is available, send prompt to summarize and put in headline_data
    - if text is not available, try to use summary from rss feed
    """

    def __init__(self, verbose: bool = False, logger: logging.Logger = None):
        self.verbose = verbose
        self.logger = logger

    def _read_text_file(self, text_path: str) -> str:
        """Helper function to read text content from file path"""
        if not text_path or text_path == '':
            return ""

        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            return content if content else ""
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to read text file {text_path}: {e}")
            return ""

    def _extract_metadata(self, html_path: str) -> Dict[str, Any]:
        """Extract metadata description and tags from HTML file"""
        def flatten(lst):
            """
            Flatten a nested list using an iterative approach with a stack.
            """
            result = []
            stack = lst.copy()

            while stack:
                item = stack.pop()
                if isinstance(item, list):
                    stack.extend(item)
                else:
                    result.append(item)

            return result[::-1]  # Reverse to maintain original order

        if not html_path or not os.path.exists(html_path):
            return {"description": "", "tags": []}

        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'html.parser')

            description = ""
            tags = []

            # Extract description from multiple sources (priority order)
            # 1. og:description
            og_desc = soup.find("meta", property="og:description")
            if og_desc and og_desc.get("content"):
                description = og_desc.get("content").strip()

            # 2. twitter:description
            if not description:
                twitter_desc = soup.find(
                    "meta", attrs={"name": "twitter:description"})
                if twitter_desc and twitter_desc.get("content"):
                    description = twitter_desc.get("content").strip()

            # 3. meta description
            if not description:
                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc and meta_desc.get("content"):
                    description = meta_desc.get("content").strip()

            # Extract tags from multiple sources
            # 1. article:tag
            article_tags = soup.find_all("meta", property="article:tag")
            for tag in article_tags:
                if tag.get("content"):
                    tags.append(tag.get("content").strip())

            # 2. article:section
            article_section = soup.find("meta", property="article:section")
            if article_section and article_section.get("content"):
                tags.append(article_section.get("content").strip())

            # 3. keywords meta tag
            keywords_tag = soup.find("meta", attrs={"name": "keywords"})
            if keywords_tag and keywords_tag.get("content"):
                keywords = [k.strip()
                            for k in keywords_tag.get("content").split(",")]
                tags.extend(keywords)

            # 4. parsely-section
            parsely_section = soup.find(
                "meta", attrs={"name": "parsely-section"})
            if parsely_section and parsely_section.get("content"):
                tags.append(parsely_section.get("content").strip())

            # 5. JSON-LD structured data keywords
            json_ld_scripts = soup.find_all(
                'script', type='application/ld+json')
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        # Handle single JSON-LD object
                        if 'keywords' in data:
                            if isinstance(data['keywords'], list):
                                tags.extend(data['keywords'])
                            elif isinstance(data['keywords'], str):
                                tags.extend([k.strip()
                                            for k in data['keywords'].split(',')])
                        if 'articleSection' in data:
                            tags.append(data['articleSection'])
                    elif isinstance(data, list):
                        # Handle array of JSON-LD objects
                        for item in data:
                            if isinstance(item, dict):
                                if 'keywords' in item:
                                    if isinstance(item['keywords'], list):
                                        tags.extend(item['keywords'])
                                    elif isinstance(item['keywords'], str):
                                        tags.extend(
                                            [k.strip() for k in item['keywords'].split(',')])
                                if 'articleSection' in item:
                                    tags.append(item['articleSection'])
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

            # Clean up tags: remove duplicates, empty strings, and normalize
            tags = flatten(tags)
            tags = list(set([tag.strip()
                        for tag in tags if tag and tag.strip()]))

            return {
                "description": description,
                "tags": tags
            }

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to extract metadata from {html_path}: {e}")
            return {"description": "", "tags": []}

    async def _extract_summaries(self, ctx, args: str) -> str:
        """Execute Step 4: Extract Summaries using AI-powered summarization"""
        step_name = "step_04_extract_summaries"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.is_step_complete(step_name):
            return "Step 4 already completed! Generated summaries."

        # Check if step 3 is completed
        if not state.is_step_complete("step_03_download_articles"):
            return "❌ Cannot execute Step 4: Step 3 (Download Articles) must be completed first."

        # Check if workflow is blocked by errors
        if state.has_errors():
            return "❌ Cannot execute Step 4: Workflow is blocked by errors. {state.workflow_status_message}"

        try:
            # Update workflow status for UI tracking
            state.start_step(step_name)
            print(f"▶ Starting Step 4: {step_name}")

            # Convert headline_data to DataFrame for processing
            headline_df = pd.DataFrame(state.headline_data)
            # Filter to AI-related articles with HTML content
            # TODO: filter to articles with text content with .str.len() > 0
            ai_articles_mask = (headline_df.get('isAI')) & (
                headline_df['text_path'].notna())
            if not ai_articles_mask.any():
                return "❌ No AI-related articles with HTML content found to summarize. Please run step 3 first."

            ai_articles_df = headline_df[ai_articles_mask].copy()

            if self.logger:
                self.logger.info(
                    f"Processing {len(ai_articles_df)} AI articles for summarization")

            # Load text content into DataFrame
            ai_articles_df['text_content'] = ai_articles_df['text_path'].apply(
                self._read_text_file)

            # Get prompt and model from Langfuse
            system_prompt, user_prompt, model = LangfuseClient(
            ).get_prompt("newsagent/extract_summaries")

            if self.logger:
                self.logger.info(f"Using model '{model}' for summarization")

            # Create LLM agent for batch summarization
            summary_agent = LLMagent(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_type=ArticleSummaryList,
                model=model,
                verbose=self.verbose,
                logger=self.logger
            )

            # Use filter_dataframe for batch summarization
            # TODO: even though we pass text_content variable
            #  always needs input_text as prompt to pass the full dataframe
            if self.logger:
                self.logger.info(
                    f"Starting summarization for {len(ai_articles_df)} articles")

            ai_articles_df['summary'] = await summary_agent.filter_dataframe(
                ai_articles_df[['id', 'text_content']],
                # This tells it to extract the list from the batch response
                item_list_field='results_list',
                # This tells it to get the 'summary' field from each ArticleSummary
                value_field='summary',
                item_id_field='id',              # This maps the responses back to the correct rows
                chunk_size=1                     # send in batches of
            )

            # Clean up text_content column
            ai_articles_df.drop('text_content', axis=1, inplace=True)

            # Calculate statistics
            articles_processed = len(ai_articles_df)
            successful_summaries = len(
                [s for s in ai_articles_df['summary'] if s and s.strip() and not s.startswith("Error")])
            summarization_errors = articles_processed - successful_summaries

            # Update headline_df with summaries
            headline_df['summary'] = ai_articles_df['summary']
            if self.logger:
                self.logger.info(
                    f"Extracting metadata from HTML files for {len(ai_articles_df)} articles")

            # Extract metadata from HTML files
            ai_articles_df['metadata'] = ai_articles_df['html_path'].apply(
                self._extract_metadata)
            # Extract description and tags from metadata
            ai_articles_df['description'] = ai_articles_df['metadata'].apply(
                lambda x: x.get('description', '') if isinstance(x, dict) else '')
            ai_articles_df['description'] = ai_articles_df['description'].fillna(
                '')
            ai_articles_df['tags'] = ai_articles_df['metadata'].apply(
                lambda x: x.get('tags', []) if isinstance(x, dict) else [])
            ai_articles_df['tags'] = ai_articles_df['tags'].fillna(
                '').apply(lambda x: [] if x == '' else x)
            ai_articles_df.drop('metadata', axis=1, inplace=True)

            # Update headline_df with description and tags
            headline_df['description'] = ai_articles_df['description']
            headline_df['tags'] = ai_articles_df['tags']

            # Make short summary

            def _get_input_text(row):
                """
                Format a news item for short summary

                Args:
                    row: DataFrame row

                Returns:
                    str: Formatted text for a news item
                """

                retval = f"{row.title}\n"

                summary = ""
                if isinstance(row.rss_summary, str) and len(row.rss_summary) > 0:
                    summary += "\n" + row.rss_summary
                elif isinstance(row.description, str) and len(row.description) > 0:
                    summary += "\n" + row.description
                if summary:
                    soup = BeautifulSoup(summary, 'html.parser')
                    summary = soup.get_text().strip()
                    retval += f"\n{summary}\n"

                if isinstance(row.summary, str) and len(row.summary) > 0:
                    retval += "\n" + row.summary

                return retval

            ai_articles_df['input_text'] = ai_articles_df.apply(
                _get_input_text, axis=1)

            system_prompt, user_prompt, model = LangfuseClient(
            ).get_prompt("newsagent/item_distiller")
            distill_agent = LLMagent(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_type=DistilledStoryList,
                model=model,
                verbose=self.verbose,
                logger=self.logger
            )
            ai_articles_df['short_summary'] = await distill_agent.filter_dataframe(
                ai_articles_df[['id', 'input_text']],
                # This tells it to extract the list from the batch response
                item_list_field='results_list',
                # This tells it to get the 'short_summary' field from each ArticleSummary
                value_field='short_summary',
                item_id_field='id',              # This maps the responses back to the correct rows
                chunk_size=1                     # send in batches of
            )
            headline_df['short_summary'] = ai_articles_df['short_summary']

            # Store updated headline data in state
            state.headline_data = headline_df.to_dict('records')

            # Complete the step
            state.complete_step(step_name)

            if self.verbose:
                print(
                    f"✅ Completed Step 4: Generated AI summaries for {successful_summaries}/{articles_processed} articles")

            status_msg = f"✅ Step 4 {step_name} completed successfully! Generated AI-powered summaries for {successful_summaries}/{articles_processed} articles."
            if summarization_errors > 0:
                status_msg += f"\n⚠️  Summarization errors: {summarization_errors}"
            status_msg += "\n💾 Summaries stored in headline DataFrame."

            # Serialize state after completing step
            state.serialize_to_db(step_name)
            return status_msg

        except Exception as e:
            state.error_step(step_name, str(e))
            return f"❌ Step 4 failed: {str(e)}"

    def create_tool(self) -> FunctionTool:
        """Create a FunctionTool instance following OpenAI Agents SDK conventions"""
        return FunctionTool(
            name="extract_summaries",
            description="Execute Step 4: Create bullet point summaries of each downloaded article. Requires Step 3 to be completed first.",
            params_json_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            on_invoke_tool=self._extract_summaries
        )


class RateArticlesTool:
    """Tool for Step 5: Rate article quality and importance
    - create a rating using prompt to compare articles according to a rubric, and ELO/Bradley-Terry model
    - optionally could use a prompt to ask if it's AI related, important and not spammy, and use log probs from prompt
    - use additional criteria like log length of article, reputation of site
    - combine to create a rating
    - ratings can later be used to inform clustering decisions and section selection
    """

    def __init__(self, verbose: bool = False, logger: logging.Logger = None):
        self.verbose = verbose
        self.logger = logger

    async def _rate_articles(self, ctx, args: str) -> str:
        """Execute Step 5: Rate Articles using persistent state"""
        step_name = "step_05_rate_articles"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.is_step_complete(step_name):
            rated_articles = [
                article for article in state.headline_data if article.get('quality_rating')]
            avg_rating = sum(article.get('quality_rating', 0)
                             for article in rated_articles) / len(rated_articles) if rated_articles else 0
            return f"Step 5 already completed! Rated {len(rated_articles)} articles with average rating {avg_rating:.1f}/10."

        # Check if step 4 is completed
        if not state.is_step_complete("step_04_extract_summaries"):
            return "❌ Cannot execute Step 5: Step 4 (Extract Summaries) must be completed first."

        # Check if workflow is blocked by errors
        if state.has_errors():
            return "❌ Cannot execute Step 5: Workflow is blocked by errors. {state.workflow_status_message}"

        try:
            # Update workflow status for UI tracking
            state.start_step(step_name)
            print(f"▶ Starting Step 5: {step_name}")

            # Get AI-related articles from persistent state
            headline_df = state.headline_df.copy()

            # Call fn_rate_articles from do_rating.py
            if self.logger:
                self.logger.info(
                    f"Rating {len(headline_df)} AI articles using fn_rate_articles")

            # dupe_count = headline_df["id"].duplicated().sum()
            # self.logger.warning(f"1 Found {dupe_count} duplicate articles")
            # simple rating - reputation + (-1 to 2) based on prob (not spammy, on-topic, important)
            rated_df = await fn_rate_articles(headline_df.copy(), logger=self.logger)
            # initial rating sort
            rated_df = rated_df.sort_values(
                "rating", ascending=False)
            # important to match id = index
            rated_df = rated_df.reset_index(drop=True)
            rated_df['id'] = rated_df.index

            # dupe_count = rated_df["id"].duplicated().sum()
            # self.logger.warning(f"2 Found {dupe_count} duplicate articles")
            # Bradley-Terry rating (like Elo)
            rated_df = await bradley_terry(rated_df, logger=self.logger)
            # dupe_count = rated_df["id"].duplicated().sum()
            # self.logger.warning(f"3 Found {dupe_count} duplicate articles")
            # scale bradley_terry rating from z-score ~(-2.5 to 2.5) to ~(-1.5 to 6)
            rated_df['bt_z'] = (rated_df['bt_z'] + 1.5) * 1.5
            rated_df['rating'] = rated_df['rating'] + rated_df['bt_z']
            # rated_df['rating'] = rated_df['rating'].clip(lower=0, upper=10)

            # Filter out low rated articles
            minimum_story_rating = 0.0
            low_rated_count = len(
                rated_df[rated_df['rating'] < minimum_story_rating])
            if self.logger:
                self.logger.info(f"Low rated articles: {low_rated_count}")
                for row in rated_df[rated_df['rating'] < minimum_story_rating].itertuples():
                    self.logger.info(
                        f"low rated article: {row.title} {row.rating}")

            # filter to minimum rating, we could filter to top n, or using rating and MMR
            rated_df = rated_df[rated_df['rating']
                                >= minimum_story_rating].copy()

            # Convert back to dict format and update state
            state.headline_data = rated_df.to_dict('records')

            # Calculate stats
            articles_rated = len(rated_df)
            avg_rating = rated_df['rating'].mean() if not rated_df.empty else 0
            high_quality_count = len(
                rated_df[rated_df['rating'] >= 7.0]) if not rated_df.empty else 0

            # Complete the step
            state.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 5: Rated {articles_rated} articles")

            # Send email with rated articles
            try:
                # Create HTML content for email
                html_items = []
                for row in rated_df.sort_values("rating", ascending=False).itertuples():
                    rating = getattr(row, 'rating', 0)
                    title = getattr(row, 'title', 'No Title')
                    url = getattr(row, 'final_url', '#')
                    site_name = getattr(row, 'site_name', 'Unknown')
                    short_summary = getattr(row, 'short_summary', '')

                    html_item = f"""
                    <div style="margin-bottom: 20px; padding: 10px; border-left: 3px solid #4CAF50;">
                        <h3 style="margin: 0 0 5px 0;">{rating:.1f} - <a href="{url}">{title}</a> - {site_name}</h3>
                        <p style="margin: 5px 0 0 0; color: #666;">{short_summary}</p>
                    </div>
                    """
                    html_items.append(html_item)

                html_content = "\n".join(html_items)

                # Create subject with date/time
                now = datetime.now()
                subject = f"AI news items - {now.strftime('%Y-%m-%d %H:%M:%S')}"

                # Send email
                send_gmail(subject, html_content)
                self.logger.info(
                    f"Sent email with {len(html_items)} rated articles")
            except Exception as e:
                self.logger.error(f"Failed to send email: {e}")

            status_msg = f"✅ Step 5 {step_name} completed successfully! Rated {articles_rated} articles with average rating {avg_rating:.1f}/10."
            status_msg += f"\n⭐ High quality articles (≥7.0): {high_quality_count}"
            status_msg += "\n💾 Ratings stored in persistent state."

            # Serialize state after completing step
            state.serialize_to_db(step_name)
            return status_msg

        except Exception as e:
            state.error_step(step_name, str(e))
            return f"❌ Step 5 failed: {str(e)}"

    def create_tool(self) -> FunctionTool:
        """Create a FunctionTool instance following OpenAI Agents SDK conventions"""
        return FunctionTool(
            name="rate_articles",
            description="Execute Step 5: Evaluate article quality and importance with ratings. Requires Step 4 to be completed first.",
            params_json_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            on_invoke_tool=self._rate_articles
        )

# review how we created categories ...
# try to extract common themes from other sections
# clean up categories to remove duplicates and merge similar themes and tighten names
# assign articles to sections or other


class ClusterByTopicTool:
    """Tool for Step 6: Cluster articles by topic

    Multi-stage topic analysis and clustering process:

    1. Extract free-form topics: Use AI to identify up to 5 distinct topics from each article summary
    2. Add frequently mentioned topics: Identify top 50 topics mentioned 3+ times from today's articles
    3. Check canonical topics: Use cheap/fast nano model to classify each article against 150+ evergreen canonical topics
    4. Clean up topics: Filter combined list (free-form + canonical) to <7 best matching topics per article
    5. Create clusters: Group articles by their final cleaned topic lists

    Stores extracted_topics, canonical_topics, and final topics_list in article data.
    Creates topic-based clusters for newsletter organization.
    """

    def __init__(self, verbose: bool = False, logger: logging.Logger = None):
        self.verbose = verbose
        self.logger = logger

    def _create_extended_summary(row):
        parts = []

        # Add title if present
        if 'title' in row and row['title']:
            parts.append(str(row['title']).strip())

        # Add description if present
        if 'description' in row and row['description']:
            parts.append(str(row['description']).strip())

        # Add topics if present (join with commas)
        if 'topics' in row and row['topics']:
            if isinstance(row['topics'], list):
                topics_str = ", ".join(str(topic).strip()
                                       for topic in row['topics'] if topic)
            else:
                topics_str = str(row['topics']).strip()
            if topics_str:
                parts.append(topics_str)

        # Add summary if present
        if pd.notna(row.get('summary')) and row.get('summary'):
            parts.append(str(row['summary']).strip())

        return "\n\n".join(parts)

    async def _get_embeddings_df(self, headline_data: pd.DataFrame, embedding_model: str = "text-embedding-3-large") -> pd.DataFrame:
        """
        Get embeddings for article summaries and return as DataFrame.

        Args:
            headline_data: DataFrame with articles containing summary column
            embedding_model: OpenAI embedding model to use

        Returns:
            DataFrame with embeddings for each extended summary
        """
        from openai import OpenAI
        from llm import paginate_df_async

        if self.logger:
            self.logger.info(
                f"Getting embeddings for {len(headline_data)} article summaries using {embedding_model}")

        # Create extended_summary column by concatenating available fields
        headline_data_copy = headline_data.copy()

        headline_data_copy['extended_summary'] = headline_data_copy.apply(
            self._create_extended_summary, axis=1)

        # Filter to articles with non-empty extended summaries
        articles_with_summaries = headline_data_copy[
            (headline_data_copy['extended_summary'].notna()) &
            (headline_data_copy['extended_summary'] != '')
        ].copy()

        if articles_with_summaries.empty:
            if self.logger:
                self.logger.warning(
                    "No articles with extended summaries found for embedding")
            return pd.DataFrame()

        all_embeddings = []
        client = OpenAI()

        # Use paginate_df_async similar to do_dedupe.py
        async for batch_df in paginate_df_async(articles_with_summaries, 25):
            text_batch = batch_df["extended_summary"].to_list()
            response = client.embeddings.create(
                input=text_batch, model=embedding_model)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        # Create DataFrame with embeddings, preserving original index
        embedding_df = pd.DataFrame(
            all_embeddings,
            index=articles_with_summaries.index
        )

        if self.logger:
            self.logger.info(
                f"Successfully generated {len(embedding_df)} embeddings with {len(embedding_df.columns)} dimensions")

        return embedding_df

    async def _free_form_extraction(self, articles_with_summaries: pd.DataFrame) -> pd.DataFrame:
        """Extract topics from article summaries using AI"""
        # Step 1: Extract topics from summaries using AI
        if self.logger:
            self.logger.info("Starting topic extraction for clustering")

        # Get prompt and model from Langfuse
        system_prompt, user_prompt, model = LangfuseClient(
        ).get_prompt("newsagent/extract_topics")

        if self.logger:
            self.logger.info(f"Using model '{model}' for topic extraction")
            self.logger.info(
                f"Processing {len(articles_with_summaries)} articles for topic extraction")

        # Create LLMagent for topic extraction
        topic_agent = LLMagent(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_type=TopicExtractionList,
            model=model,
            verbose=self.verbose,
            logger=self.logger
        )

        # Extract topics using filter_dataframe
        articles_with_summaries['extracted_topics'] = await topic_agent.filter_dataframe(
            articles_with_summaries[['id', 'input_text']],
            value_field='topics_list',
            item_list_field='results_list',
            item_id_field='id',
            chunk_size=10
        )
        # Handle NaN values in extracted_topics
        articles_with_summaries['extracted_topics'] = articles_with_summaries['extracted_topics'].fillna('').apply(
            lambda x: list() if not isinstance(x, list) else x
        )

        if self.logger:
            topics_extracted = articles_with_summaries['extracted_topics'].apply(
                lambda x: len(x) if isinstance(x, list) else 0).sum()
            self.logger.info(
                f"Successfully extracted {topics_extracted} total topics across articles")

        return articles_with_summaries

    async def _classify_canonical_topic(self, headline_df: pd.DataFrame, topic: str) -> List[bool]:
        """
        Classify all summaries against a single canonical topic

        Args:
            headline_df: DataFrame with articles containing summaries
            topic: Single canonical topic to classify against

        Returns:
            List of boolean values indicating relevance to the topic
        """
        # Get prompt and model from Langfuse
        system_prompt, user_prompt, model = LangfuseClient().get_prompt(
            "newsagent/canonical_topic")

        # Create LLMagent for canonical topic classification
        canonical_agent = LLMagent(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_type=CanonicalTopicClassificationList,
            model=model,
            verbose=False,  # too much output , always false
            logger=self.logger
        )

        # Use filter_dataframe to classify against the canonical topic
        relevance_series = await canonical_agent.filter_dataframe(
            headline_df[['id', 'input_text']].copy(),
            value_field='relevant',
            item_list_field='results_list',
            item_id_field='id',
            input_vars={'topic': topic},
            chunk_size=10
        )

        return topic, relevance_series.tolist()

    async def _canonical_topic_extraction(self, articles_with_summaries: pd.DataFrame) -> pd.DataFrame:
        """Extract canonical topics from article summaries using AI"""
        # Classify against canonical topics
        if self.logger:
            self.logger.info(
                f"Starting canonical topic classification for {len(CANONICAL_TOPICS)} topics")

        # Create all canonical topic classification tasks
        canonical_tasks = [
            self._classify_canonical_topic(articles_with_summaries, topic)
            for topic in CANONICAL_TOPICS
        ]

        # Run all canonical topic classifications concurrently
        canonical_results = await asyncio.gather(*canonical_tasks, return_exceptions=True)
        # Initialize canonical_topics as list of empty lists
        canonical_topics_lists = [list()
                                  for _ in range(len(articles_with_summaries))]

        # Process canonical results
        for result in canonical_results:
            if isinstance(result, Exception):
                if self.logger:
                    self.logger.warning(
                        f"Topic classification failed: {result}")
                continue

            topic, relevance_list = result
            if isinstance(relevance_list, list):
                for idx, is_relevant in enumerate(relevance_list):
                    if is_relevant:
                        canonical_topics_lists[idx].append(topic)

        # Assign to canonical_topics column
        articles_with_summaries['canonical_topics'] = canonical_topics_lists

        if self.logger:
            total_canonical_matches = sum(
                len(topics) for topics in articles_with_summaries['canonical_topics'])
            self.logger.info(
                f"Canonical topic classification complete: {total_canonical_matches} total topic matches")

        return articles_with_summaries

    async def _cleanup_topics(self, headline_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up and filter combined topics using AI to select best 3-7 topics

        Args:
            headline_df: DataFrame with articles containing extracted_topics, canonical_topics, and summaries

        Returns:
            DataFrame with cleaned topics_list column
        """
        system_prompt, user_prompt, model = LangfuseClient(
        ).get_prompt("newsagent/topic_cleanup")

        if self.logger:
            self.logger.info(
                f"Starting topic cleanup for {len(headline_df)} articles")

        # Combine all topic sources into a single column removing duplicates
        headline_df['all_topics'] = headline_df.apply(
            lambda row: list(set(
                (row.get('tags', []) if isinstance(row.get('tags'), list) else []) +
                (row.get('extracted_topics', []) if isinstance(row.get('extracted_topics'), list) else []) +
                (row.get('canonical_topics', []) if isinstance(
                    row.get('canonical_topics'), list) else [])
            )), axis=1
        )

        # Create LLMagent for topic cleanup
        cleanup_agent = LLMagent(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_type=TopicExtractionList,
            model=model,
            verbose=self.verbose,
            logger=self.logger
        )

        # Use filter_dataframe to clean up topics
        headline_df['topics'] = await cleanup_agent.filter_dataframe(
            headline_df[['id', 'summary', 'all_topics']],
            value_field='topics_list',
            item_list_field='results_list',
            item_id_field='id'
        )

        return headline_df

    async def _cluster_by_topic(self, ctx, args: str) -> str:
        """Execute Step 6: Cluster By Topic using persistent state"""
        step_name = "step_06_cluster_by_topic"
        # todo: combine title and description and summary and tags for topic extraction
        # todo: show list of common topics

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.is_step_complete(step_name):
            cluster_count = len(state.clusters)
            total_articles = sum(len(articles)
                                 for articles in state.clusters.values())
            return f"Step 6 already completed! Created {cluster_count} topic clusters with {total_articles} articles."

        # Check if step 5 is completed
        if not state.is_step_complete("step_05_rate_articles"):
            return "❌ Cannot execute Step 6: Step 5 (Rate Articles) must be completed first."

        # Check if workflow is blocked by errors
        if state.has_errors():
            return f"❌ Cannot execute Step 6: Workflow is blocked by errors. {state.workflow_status_message}"

        try:
            # Update workflow status for UI tracking
            state.start_step(step_name)
            print(f"▶ Starting Step 6: {step_name}")

            # Get articles with summaries from persistent state
            headline_df = pd.DataFrame(state.headline_data)
            articles_with_summaries = headline_df.loc[
                (headline_df['isAI']) &
                headline_df['summary'].notna() &
                (headline_df['summary'] != '')]

            if articles_with_summaries.empty:
                return "❌ No summarized articles found to cluster. Please run step 4 first."

            # Clear existing clusters
            state.clusters = {}

            def _get_input_text(row):
                retval = row['title']
                if isinstance(row['rss_summary'], str) and len(row['rss_summary']) > 0:
                    retval += "\n" + row['rss_summary']
                if isinstance(row['description'], str) and len(row['description']) > 0:
                    retval += "\n" + row['description']
                if isinstance(row['summary'], str) and len(row['summary']) > 0:
                    retval += "\n" + row['summary']
                return retval

            articles_with_summaries['input_text'] = articles_with_summaries.apply(
                _get_input_text, axis=1)

            articles_with_summaries = await self._free_form_extraction(articles_with_summaries)

            # here we could make a list of all topics and grab frequently mentioned ones and add to canonical topics
            # # Collect all topics from articles with summaries
            # all_topics = []
            # for idx, row in headline_df.iterrows():
            #     if row.get('topics_list') and isinstance(row.get('topics_list'), list):
            #         all_topics.extend([topic.strip() for topic in row.get('topics_list', []) if topic.strip()])

            # # Use Counter to find frequently mentioned topics (appearing 3+ times)
            # topic_counter = Counter(all_topics)

            # # Get topics mentioned 3+ times, sorted by frequency, take top 50
            # frequent_topics = [
            #     topic for topic, count in topic_counter.most_common()
            #     if count >= 3
            # ][:50]
            # if self.logger:
            #     total_frequent_mentions = sum(topic_counter[topic] for topic in frequent_topics)
            #     self.logger.info(f"Found {len(frequent_topics)} frequently mentioned topics (3+ times) from {len(all_topics)} total topic instances")
            #     self.logger.info(f"Total unique topics: {len(unique_topics)}")
            #     self.logger.info(f"Top frequent topics represent {total_frequent_mentions} mentions")
            #     if frequent_topics:
            #         # Show top 10 with their counts
            #         top_topics_with_counts = [(topic, topic_counter[topic]) for topic in frequent_topics[:10]]
            #         self.logger.info(f"Top frequent topics: {top_topics_with_counts}")
            #         if len(frequent_topics) > 10:
            #             self.logger.info(f"... and {len(frequent_topics) - 10} more")

            articles_with_summaries = await self._canonical_topic_extraction(articles_with_summaries)

            articles_with_summaries = await self._cleanup_topics(articles_with_summaries)

            headline_df['topics'] = articles_with_summaries['topics']
            if 'tags' in headline_df.columns:
                headline_df = headline_df.drop(columns=['tags'])

            headline_df = await do_clustering(headline_df, logger=self.logger)

            # re-index
            headline_df.reset_index(drop=True, inplace=True)
            headline_df['id'] = headline_df.index

            # Complete the step
            state.headline_data = headline_df.to_dict('records')
            state.complete_step(step_name)

            # if self.verbose:
            #     print(f"✅ Completed Step 5: Created {total_clusters} topic clusters")

            # Calculate canonical topic stats
            # total_canonical_matches = sum(len(article.get('canonical_topics', [])) for article in state.headline_data)

            status_msg = f"✅ Step 6 {step_name} completed successfully! Organized {len(headline_df)} articles into topic clusters."
            # status_msg += f"\n📊 Cluster coherence score: {cluster_coherence_score:.1%}"
            # status_msg += f"\n🔄 Frequent topics found: {len(state.common_topics)} (top 50 topics appearing 3+ times)"
            # status_msg += f"\n🏛️ Canonical topic matches: {total_canonical_matches} across {len(CANONICAL_TOPICS)} canonical topics"
            # status_msg += f"\n🏷️ Topics: {', '.join(state.clusters.keys())}"
            # status_msg += f"\n💾 Clusters, common topics, and canonical classifications stored in persistent state."

            # Serialize state after completing step
            state.serialize_to_db(step_name)
            return status_msg

        except Exception as e:
            state.error_step(step_name, str(e))
            return f"❌ Step 6 failed: {str(e)}"

    def create_tool(self) -> FunctionTool:
        """Create a FunctionTool instance following OpenAI Agents SDK conventions"""
        return FunctionTool(
            name="cluster_by_topic",
            description="Execute Step 6: Group articles by thematic topics using clustering. Requires Step 5 to be completed first.",
            params_json_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            on_invoke_tool=self._cluster_by_topic
        )


class SelectSectionsTool:
    """Tool for Step 7: Select newsletter sections
    select all stories with a minimum rating
    send stories and a prompt to select major themes
    send a prompt with all the themes and refine it to 6-12 themes
    send a prompt to all stories and assign to a final theme or other
    """

    def __init__(self, verbose: bool = False, logger: logging.Logger = None):
        self.verbose = verbose
        self.logger = logger

    async def _select_sections(self, ctx, args: str) -> str:
        """Execute Step 7: Select Sections using persistent state"""
        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        step_name = "step_07_select_sections"
        if state.is_step_complete(step_name):
            section_count = len(state.newsletter_section_text)
            return f"Step 7 already completed! Created {section_count} newsletter sections."

        # Check if step 6 is completed
        if not state.is_step_complete("step_06_cluster_by_topic"):
            return "❌ Cannot execute Step 7: Step 6 (Cluster By Topic) must be completed first."

        # Check if workflow is blocked by errors
        if state.has_errors():
            return "❌ Cannot execute Step 7: Workflow is blocked by errors. {state.workflow_status_message}"

        def _get_input_text(row):
            """
            Format a news item for assignment to categories

            Args:
                row: DataFrame row

            Returns:
                str: Formatted text for a news item
            """

            retval = f"{row.title} - {row.site_name}\n"

            retval += f"\nRating: {row.rating:.1f}\n"

            if hasattr(row, 'topics') and row.topics:
                topics = ", ".join(row.topics)
                retval += f"\nTopics: {topics}\n"

            summary = ""
            if isinstance(row.rss_summary, str) and len(row.rss_summary) > 0:
                summary += "\n" + row.rss_summary
            elif isinstance(row.description, str) and len(row.description) > 0:
                summary += "\n" + row.description
            if summary:
                soup = BeautifulSoup(summary, 'html.parser')
                summary = soup.get_text().strip()
                retval += f"\n{summary}\n"

            if isinstance(row.summary, str) and len(row.summary) > 0:
                retval += "\n" + row.summary

            retval += "\n---\n"

            return retval

        try:
            # Update workflow status for UI tracking
            state.start_step(step_name)
            print(f"▶ Starting Step 7: {step_name}")
            # Get rated articles from persistent state
            headline_df = state.headline_df.copy()

            self.logger.info("Free form categorization of articles")

            system_prompt, user_prompt, model = LangfuseClient().get_prompt(
                "newsagent/cat_proposal")

            cat_proposal_agent = LLMagent(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_type=TopicCategoryList,
                model=model,
                verbose=self.verbose,
                logger=self.logger
            )

            headline_df["input_text"] = headline_df.apply(
                _get_input_text, axis=1)
            input_text = "\n".join(headline_df["input_text"].tolist())

            suggested_cats = await cat_proposal_agent.run_prompt(input_text=input_text)
            suggested_cats_list = [cat for cat in suggested_cats.items]

            # combine with hdbscan clusters and remove duplicates
            initial_cats = list(
                set(headline_df["cluster_name"].to_list() + suggested_cats_list))

            self.logger.info(f"Cleaning up initial categories: {initial_cats}")

            system_prompt, user_prompt, model = LangfuseClient(
            ).get_prompt("newsagent/cat_cleanup")

            cat_cleanup_agent = LLMagent(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_type=TopicCategoryList,
                model=model,
                verbose=self.verbose,
                logger=self.logger
            )
            response = await cat_cleanup_agent.run_prompt(input_text="\n".join(initial_cats))
            final_cats = [cat for cat in response.items]
            final_cats_str = "\n".join(sorted(final_cats))
            self.logger.info(f"Final categories: {final_cats_str}")

            # loop over items and assign to cats
            system_prompt, user_prompt, model = LangfuseClient(
            ).get_prompt("newsagent/cat_assignment")

            cat_assignment_agent = LLMagent(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_type=TopicHeadline,
                model=model,
                verbose=self.verbose,
                logger=self.logger
            )

            async def assign_topic(idx, input_text, topics_str=None):
                topics = topics_str if topics_str is not None else final_cats_str
                assigned_cat = await cat_assignment_agent.run_prompt(topics=topics,
                                                                     input_text=input_text)
                return (idx, input_text, assigned_cat.topic_title)

            tasks = [assign_topic(row.id, row.input_text)
                     for row in headline_df.itertuples()]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            templist = [t for t in results if not isinstance(t, Exception)]

            catdf = pd.DataFrame(templist, columns=["id", "input_text", "cat"])
            catdf["cat"] = catdf["cat"].fillna("Other")
            catdf.loc[catdf["cat"] == "None", "cat"] = "Other"

            headline_df = headline_df.merge(
                catdf[["id", "cat"]], on="id", how="left")

            # get unique cluster names and sort them
            cluster_df = headline_df["cat"].value_counts().reset_index()
            cluster_df.columns = ["cat", "count"]
            self.logger.info(
                f"Assigned articlles to {len(cluster_df)} categories")
            self.logger.info(
                f"Cluster counts: {cluster_df.to_dict(orient='records')}")

            # dedupe articles
            self.logger.info("Deduping articles")
            system_prompt, user_prompt, model = LangfuseClient().get_prompt(
                "newsagent/dedupe_articles")

            dedupe_agent = LLMagent(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_type=DupeRecordList,
                model=model,
                verbose=self.verbose,
                logger=self.logger
            )

            async def run_dedupe(input_text):
                response = await dedupe_agent.run_prompt(input_text=input_text)
                return pd.DataFrame([(dr.id, dr.dupe_id) for dr in response.results_list], columns=["id", "dupe_id"])

            tasks = []
            for cat in cluster_df["cat"]:
                tmpdf = headline_df.loc[headline_df["cat"] == cat].sort_values(
                    "rating", ascending=False).copy()
                if len(tmpdf) > 1:  # at least 2 to dedupe
                    self.logger.info(
                        f"Deduping cat: {cat} ({len(tmpdf)} items)")
                    input_text = tmpdf.loc[tmpdf["cat"] == cat][[
                        "id", "extended_summary"]].to_json()
                    tasks.append(run_dedupe(input_text))

            deduped_dfs = await asyncio.gather(*tasks)

            # concatenate deduped_dfs into a single df
            deduped_df = pd.concat(deduped_dfs)
            headline_df = pd.merge(
                headline_df, deduped_df[["id", "dupe_id"]], on="id", how="left")
            # count number of rows in aidf where dupe_id is >0 and group by dupe_id
            dupe_counts = headline_df.loc[headline_df['dupe_id'] > 0].groupby(
                'dupe_id').size()
            self.logger.info(dupe_counts)
            # for each dupe_id in dupe_counts, add the count to the rating of that id
            for dupe_id in dupe_counts.index:
                headline_df.loc[headline_df['id'] == dupe_id,
                                'rating'] += dupe_counts[dupe_id]
            self.logger.info(f"Deduped articles: {len(headline_df)}")
            dupe_df = headline_df.loc[headline_df['dupe_id'] > 0]
            self.logger.info(
                f"Duplicating {len(dupe_df)} articles: {dupe_df.to_dict(orient='records')}")
            # drop rows where dupe_id is >= 0 (keep rows where dupe_id is -1, ie unique)
            headline_df = headline_df.loc[headline_df['dupe_id'] < 0]

            # Handle singleton categories (categories with only 1 article)
            self.logger.info("Handling singleton categories")
            cat_counts_df = headline_df["cat"].value_counts().reset_index()
            cat_counts_df.columns = ["cat", "count"]

            # Split into singletons (count == 1) and non-singletons (count > 1)
            singletons = cat_counts_df.loc[cat_counts_df["count"] == 1, "cat"].tolist(
            )
            nonsingletons = cat_counts_df.loc[cat_counts_df["count"] > 1, "cat"].tolist(
            )

            # Ensure 'Other' is in nonsingletons and not in singletons
            if "Other" in singletons:
                singletons.remove("Other")
            if "Other" not in nonsingletons:
                nonsingletons.append("Other")

            self.logger.info(
                f"Singleton categories ({len(singletons)}): {singletons}")
            self.logger.info(
                f"Non-singleton categories ({len(nonsingletons)}): {nonsingletons}")

            # Reassign singleton articles to non-singleton categories
            if singletons and nonsingletons:
                singleton_rows = headline_df.loc[headline_df["cat"].isin(
                    singletons)]

                if len(singleton_rows) > 0:
                    self.logger.info(
                        f"Reassigning {len(singleton_rows)} singleton articles to non-singleton categories")

                    nonsingletons_str = "\n".join(sorted(nonsingletons))

                    reassign_tasks = [
                        assign_topic(row.id, row.input_text, nonsingletons_str)
                        for row in singleton_rows.itertuples()
                    ]
                    reassign_results = await asyncio.gather(*reassign_tasks, return_exceptions=True)
                    reassign_valid = [
                        r for r in reassign_results if not isinstance(r, Exception)]

                    # Create dataframe and update categories
                    reassign_df = pd.DataFrame(
                        reassign_valid, columns=["id", "input_text", "cat"])
                    reassign_df["cat"] = reassign_df["cat"].fillna("Other")
                    reassign_df.loc[reassign_df["cat"]
                                    == "None", "cat"] = "Other"

                    # Update the headline_df with reassigned categories
                    for _, row in reassign_df.iterrows():
                        headline_df.loc[headline_df['id'] ==
                                        row['id'], 'cat'] = row['cat']

                    self.logger.info(
                        f"Reassigned {len(reassign_valid)} singleton articles")

                    # Log final category counts
                    final_cat_counts = headline_df["cat"].value_counts(
                    ).reset_index()
                    final_cat_counts.columns = ["cat", "count"]
                    self.logger.info(
                        f"Final category counts: {final_cat_counts.to_dict(orient='records')}")

            state.headline_data = headline_df.to_dict('records')

            # Complete the step
            state.complete_step(step_name)

            cat_article_counts = headline_df.groupby("cat").count()[
                "input_text"]
            state.workflow_status_message = f"Categories and article counts:\n{cat_article_counts.to_string()}"

            if self.verbose:
                print(f"✅ Completed Step 7: {state.workflow_status_message}")

            # Serialize state after completing step
            state.serialize_to_db(step_name)
            return state.workflow_status_message

        except Exception as e:
            state.error_step(step_name, str(e))
            return f"❌ Step 7 failed: {str(e)}"

    def create_tool(self) -> FunctionTool:
        """Create a FunctionTool instance following OpenAI Agents SDK conventions"""
        return FunctionTool(
            name="select_sections",
            description="Execute Step 7: Organize articles into newsletter sections based on topics and ratings. Requires Step 6 to be completed first.",
            params_json_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            on_invoke_tool=self._select_sections
        )


class DraftSectionsTool:
    """Tool for Step 8: Draft section content
    for each section, send a prompt to draft content
    send section to a prompt to draft a title that is on topic and engaging,funny, punny, etc.
    send result to a second prompt to check format and rewrite
    use critic-optimizer agentic pattern
    """

    def __init__(self, verbose: bool = False, logger: logging.Logger = None):
        self.verbose = verbose
        self.logger = logger

    async def _draft_sections(self, ctx, args: str) -> str:
        """Execute Step 8: Draft Sections using persistent state"""
        step_name = "step_08_draft_sections"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.is_step_complete(step_name):
            section_count = len(state.newsletter_section_text)
            total_words = sum(len(content.split())
                              for content in state.newsletter_section_text.values())
            return f"Step 8 already completed! Drafted {section_count} sections with {total_words} total words."

        # Check if step 7 is completed
        if not state.is_step_complete("step_07_select_sections"):
            return "❌ Cannot execute Step 8: Step 7 (Select Sections) must be completed first."

        # Check if workflow is blocked by errors
        if state.has_errors():
            return f"❌ Cannot execute Step 8: Workflow is blocked by errors. {state.workflow_status_message}"

        try:
            # Update workflow status for UI tracking
            state.start_step(step_name)
            print(f"▶ Starting Step 8: {step_name}")

            # Get headline data
            headline_df = pd.DataFrame(state.headline_data)

            if headline_df.empty:
                return "❌ No headlines found to draft sections. Please run previous steps first."

            # Get unique categories
            categories = headline_df['cat'].unique().tolist()
            categories = [cat for cat in categories if cat != "Other"]

            if not categories:
                return "❌ No categories found in headlines. Please run step 7 first."

            self.logger.info(
                f"Drafting sections for {len(categories)} categories")

            # Create write_section agent
            write_section_system_prompt, write_section_user_prompt, model = \
                LangfuseClient().get_prompt("newsagent/write_section")

            write_section_agent = LLMagent(
                system_prompt=write_section_system_prompt,
                user_prompt=write_section_user_prompt,
                output_type=Section,
                model=model,
                verbose=self.verbose,
                logger=self.logger
            )

            async def draft_section(cat, agent):
                """Draft a section for a given category"""
                # Get articles for this category, sorted by rating, convert to JSON
                cat_df = headline_df.loc[headline_df["cat"] == cat].sort_values(
                    "rating", ascending=False)

                input_text = cat_df[["rating", "short_summary", "site_name", "final_url"]].rename(columns={"short_summary": "summary", "final_url": "url"}).to_json(
                    orient="records")

                # Call the LLM to draft the section
                response = await agent.run_prompt(input_text=input_text)

                return (cat, response)

            # Draft all sections asynchronously
            draft_tasks = [draft_section(cat, write_section_agent)
                           for cat in categories]
            draft_results = await asyncio.gather(*draft_tasks, return_exceptions=True)

            # Filter out exceptions and store results
            sections_drafted = 0
            for result in draft_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error drafting section: {result}")
                    continue

                cat, content = result
                state.newsletter_section_obj[cat] = content
                state.newsletter_section_text[cat] = content
                sections_drafted += 1

            # Complete the step
            state.complete_step(step_name)

            total_words = sum(len(content.split())
                              for content in state.newsletter_section_text.values())
            state.workflow_status_message = f"Drafted {sections_drafted} sections with {total_words} total words"

            if self.verbose:
                print(f"✅ Completed Step 8: {state.workflow_status_message}")

            # Serialize state after completing step
            state.serialize_to_db(step_name)
            return state.workflow_status_message

        except Exception as e:
            state.error_step(step_name, str(e))
            return f"❌ Step 8 failed: {str(e)}"

    def create_tool(self) -> FunctionTool:
        """Create a FunctionTool instance following OpenAI Agents SDK conventions"""
        return FunctionTool(
            name="draft_sections",
            description="Execute Step 8: Write engaging content for each newsletter section. Requires Step 7 to be completed first.",
            params_json_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            on_invoke_tool=self._draft_sections
        )


class FinalizeNewsletterTool:
    """Tool for Step 9: Finalize complete newsletter
    assemble sections.
    send full newsletter to a critic optimizer loop to refine and finalize and choose overall subject line"""

    def __init__(self, verbose: bool = False, logger: logging.Logger = None):
        self.verbose = verbose
        self.logger = logger

    async def _finalize_newsletter(self, ctx, args: str) -> str:
        """Execute Step 9: Finalize Newsletter using persistent state"""
        step_name = "step_09_finalize_newsletter"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.is_step_complete(step_name):
            newsletter_length = len(
                state.final_newsletter.split()) if state.final_newsletter else 0
            sections_count = len(state.newsletter_section_text)
            return f"Step 9 already completed! Newsletter finalized with {sections_count} sections and {newsletter_length} words."

        # Check if step 8 is completed
        if not state.is_step_complete("step_08_draft_sections"):
            return "❌ Cannot execute Step 9: Step 8 (Draft Sections) must be completed first."

        # Check if workflow is blocked by errors
        if state.has_errors():
            return f"❌ Cannot execute Step 9: Workflow is blocked by errors. {state.workflow_status_message}"

        try:
            # Update workflow status for UI tracking
            state.start_step(step_name)
            print(f"▶ Starting Step 9: {step_name}")

            # Get drafted sections from persistent state
            if not state.newsletter_section_text:
                return "❌ No drafted sections found to finalize. Please run step 8 first."

            # Create the final newsletter by combining all sections
            today = datetime.now().strftime("%B %d, %Y")

            newsletter_content = f"# AI News Digest - {today}\n\n"
            newsletter_content += "*Curated insights from the latest in artificial intelligence*\n\n"
            newsletter_content += "---\n\n"

            # Add table of contents
            newsletter_content += "## Table of Contents\n\n"
            for i, section_name in enumerate(state.newsletter_section_text.keys(), 1):
                newsletter_content += f"{i}. [{section_name}](#{section_name.lower().replace(' ', '-').replace('&', 'and')})\n"
            newsletter_content += "\n---\n\n"

            # Add each section content
            for section_name, section_content_text in state.newsletter_section_text.items():
                newsletter_content += section_content_text
                newsletter_content += "\n---\n\n"

            # Add footer
            newsletter_content += "## About This Newsletter\n\n"
            newsletter_content += "This AI News Digest was automatically curated using our intelligent newsletter agent. "
            newsletter_content += f"We analyzed {len(state.headline_data)} articles from {len(set(a.get('source', '') for a in state.headline_data))} sources "
            newsletter_content += "to bring you the most relevant AI developments.\n\n"
            newsletter_content += f"*Generated on {today}*\n"

            # Store the final newsletter
            state.final_newsletter = newsletter_content

            # Calculate final stats
            newsletter_length = len(newsletter_content.split())
            sections_included = len(drafted_sections)

            # Mock quality score based on content metrics
            base_quality = 7.0
            if sections_included >= 4:
                base_quality += 0.5
            if newsletter_length >= 2000:
                base_quality += 0.5
            if newsletter_length >= 3000:
                base_quality += 0.5
            final_quality_score = min(10.0, base_quality)

            # Complete the step and mark workflow as complete
            state.complete_step(step_name)

            if self.verbose:
                print(
                    f"✅ Completed Step 9: Finalized newsletter ({newsletter_length} words)")

            status_msg = f"🎉 Step 9 {step_name} completed successfully! Newsletter finalized with {sections_included} sections and {newsletter_length} words."
            status_msg += f"\n⭐ Quality score: {final_quality_score:.1f}/10"
            status_msg += "\n📰 Complete newsletter stored in persistent state"
            status_msg += "\n✅ Workflow complete! All 9 steps finished successfully."

            # Serialize state after completing step
            state.serialize_to_db(step_name)
            return status_msg

        except Exception as e:
            state.error_step(step_name, str(e))
            return f"❌ Step 9 failed: {str(e)}"

    def create_tool(self) -> FunctionTool:
        """Create a FunctionTool instance following OpenAI Agents SDK conventions"""
        return FunctionTool(
            name="finalize_newsletter",
            description="Execute Step 9: Combine all sections into the final newsletter with formatting and polish. Requires Step 8 to be completed first.",
            params_json_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            on_invoke_tool=self._finalize_newsletter
        )


class NewsletterAgent(Agent[NewsletterAgentState]):
    """Newsletter agent with persistent state and workflow tools"""

    def __init__(self,
                 session_id: str = "newsletter_agent",
                 state: Optional[NewsletterAgentState] = None,
                 verbose: bool = False,
                 logger: logging.Logger = None,
                 timeout: float = 300.0,
                 **kwargs):
        """
        Initialize the NewsletterAgent with persistent state

        Args:
            session_id: Unique identifier for the session (for persistence)
            state: Optional NewsletterAgentState to use. If None, creates new or loads from session
            verbose: Enable verbose logging
            logger: Optional logger instance (creates one if None)
            timeout: Timeout in seconds for OpenAI API calls (default: 300.0)
        """
        # Set up OpenAI client with custom timeout before initializing parent
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            set_default_openai_client(AsyncOpenAI(
                api_key=api_key,
                timeout=timeout
            ))
        self.session = SQLiteSession(session_id, "newsletter_agent.db")
        self.verbose = verbose
        self.logger = logger or setup_logging(session_id, LOGDB)

        # Initialize state - use provided state or load from session
        if state is not None:
            self.state = state
            if self.verbose:
                self.logger.info(
                    f"Using provided state with {len(state.headline_data)} articles")
        else:
            self.logger.info("Trying to load state from existing session")

            # Try to load existing state from session
            try:
                # The OpenAI Agents SDK Runner.run() saves context to session automatically
                # We need to check if there's existing context for this session
                if hasattr(self.session, 'get_context'):
                    self.logger.info("Using existing session get_context")
                    existing_context = self.session.get_context()
                elif hasattr(self.session, 'load_context'):
                    self.logger.info("Using existing session load_context")
                    existing_context = self.session.load_context()
                else:
                    self.logger.info("fallback to getattr")
                    existing_context = getattr(self.session, '_context', None)

                if existing_context and isinstance(existing_context, NewsletterAgentState):
                    self.state = existing_context
                    if self.verbose:
                        self.logger.info(
                            f"Restored state from session '{session_id}' with {len(self.state.headline_data)} articles at step {self.state.get_current_step()}")
                else:
                    self.logger.info(
                        "No existing context found, create new state")
                    # No existing context found, create new state
                    self.state = NewsletterAgentState()
                    if self.verbose:
                        self.logger.info(
                            f"Created new NewsletterAgentState for session '{session_id}' (no existing context found)")

            except Exception as e:
                # Fallback to new state if session loading fails
                self.state = NewsletterAgentState()
                if self.verbose:
                    self.logger.warning(
                        f"Could not load session state for '{session_id}': {e}. Created new NewsletterAgentState.")

        # System prompt that guides tool selection based on workflow status
        system_prompt = """
You are an AI newsletter writing agent that executes a 9-step workflow process using tools with persistent state.

WORKFLOW OVERVIEW:
1. Step 1: Gather URLs - Collect headlines and URLs from various sources
2. Step 2: Filter URLs - Filter headlines to AI-related content only
3. Step 3: Download Articles - Fetch full article content from URLs
4. Step 4: Extract Summaries - Create bullet point summaries of each article
5. Step 5: Rate Articles - Evaluate article quality and importance
6. Step 6: Cluster By Topic - Group articles by thematic topics
7. Step 7: Select Sections - Organize articles into newsletter sections
8. Step 8: Draft Sections - Write content for each section
9. Step 9: Finalize Newsletter - Combine sections into final newsletter

WORKFLOW RESUME LOGIC:
- You maintain persistent state between runs and can resume from any step
- ALWAYS start by checking workflow status to understand current progress
- If current_step >= 1, you can resume from any completed step forward
- Steps are idempotent - if a step is already completed, tools will return cached results
- When resuming, automatically continue from the next incomplete step

INSTRUCTIONS:
- ALWAYS start by checking the current workflow status using check_workflow_status
- Use inspect_state tool to examine detailed state data when debugging
- Execute workflow steps in the correct order using the appropriate tools
- Each step has prerequisites - only execute a step if the previous step is completed
- If a user asks to "run all steps" or "create the newsletter", execute all remaining steps in sequence
- If a user asks for a specific step, execute only that step (if prerequisites are met)
- If a user asks to "resume" or "continue", start from the next incomplete step
- Always check status between steps to ensure proper sequencing
- Your state persists between sessions - you can resume work from where you left off

TOOL SELECTION STRATEGY:
1. First, always use check_workflow_status to understand current state and progress
2. If resuming, identify the next step that needs to be executed
3. Use the appropriate tool for that step
4. After each step, check status again to confirm progress
5. Continue until workflow is complete or user request is fulfilled

RESUME EXAMPLES:
- If current_step=3, next step is step 4 (Extract Summaries)
- If current_step=7, next step is step 8 (Draft Sections)
- If current_step=9, workflow is complete - no further steps needed

Remember: Your state is persistent. You can safely resume from any point. Never skip steps or execute them out of order.
"""

        super().__init__(
            name="NewsletterAgent",
            instructions=system_prompt,
            model="gpt-5-mini",
            tools=[
                WorkflowStatusTool(self.logger).create_tool(),
                StateInspectionTool(self.verbose, self.logger).create_tool(),
                GatherUrlsTool(self.verbose, self.logger).create_tool(),
                FilterUrlsTool(self.verbose, self.logger).create_tool(),
                DownloadArticlesTool(self.verbose, self.logger).create_tool(),
                ExtractSummariesTool(self.verbose, self.logger).create_tool(),
                ClusterByTopicTool(self.verbose, self.logger).create_tool(),
                RateArticlesTool(self.verbose, self.logger).create_tool(),
                SelectSectionsTool(self.verbose, self.logger).create_tool(),
                DraftSectionsTool(self.verbose, self.logger).create_tool(),
                FinalizeNewsletterTool(
                    self.verbose, self.logger).create_tool(),
            ],
            **kwargs
        )

        # Create tool dictionary
        self._tool_dict = {tool.name: tool for tool in self.tools}

        if self.verbose:
            print(
                "Initialized NewsletterAgent with persistent state and 9-step workflow")
            print(f"Session ID: {session_id}")

    def get_current_state_summary(self) -> str:
        """Get a summary of the current state for debugging and monitoring

        Returns:
            Formatted string with key state information including session ID,
            current step, progress percentage, and data counts
        """
        try:
            # Basic session and workflow info
            summary_lines = [
                f"Session ID: {self.session.session_id}",
                f"Current Step: {self.state.get_current_step()}",
                f"Workflow Status: {self.state.workflow_status.value}",
                f"Progress: {self.state.get_progress_percentage():.1f}%",
                f"Workflow Complete: {self.state.all_complete()}",
            ]

            # Add error info if present
            if self.state.has_errors():
                summary_lines.append(
                    f"⚠️  Has Errors: {self.state.workflow_status_message}")

            # Data summary
            summary_lines.extend([
                "",
                "DATA SUMMARY:",
                f"  Total articles: {len(self.state.headline_data)}",
            ])

            if self.state.headline_data:
                ai_related = sum(
                    1 for a in self.state.headline_data if a.get('isAI') is True)
                with_summaries = sum(
                    1 for a in self.state.headline_data if a.get('summary'))
                with_topics = sum(1 for a in self.state.headline_data if a.get(
                    'extracted_topics') or a.get('canonical_topics'))
                sources = len(set(a.get('source', 'Unknown')
                              for a in self.state.headline_data))

                summary_lines.extend([
                    f"  AI-related: {ai_related}",
                    f"  With summaries: {with_summaries}",
                    f"  With topics: {with_topics}",
                    f"  Sources: {sources}",
                ])

            # Processing results
            summary_lines.extend([
                "",
                "PROCESSING RESULTS:",
                f"  Topic clusters: {len(self.state.clusters)}",
                f"  Newsletter sections: {len(self.state.newsletter_section_text)}",
                f"  Final newsletter: {'Generated' if self.state.final_newsletter else 'Not created'}",
            ])

            # Common topics if available
            if hasattr(self.state, 'common_topics') and self.state.common_topics:
                summary_lines.extend([
                    "",
                    f"COMMON TOPICS ({len(self.state.common_topics)}):",
                    f"  {', '.join(self.state.common_topics[:5])}" + (
                        "..." if len(self.state.common_topics) > 5 else "")
                ])

            return "\n".join(summary_lines)

        except Exception as e:
            return f"Error generating state summary: {str(e)}"

    async def run_tool_direct(self, tool_name: str, tool_args: str = "") -> Any:
        """Run a specific tool directly by name, bypassing LLM

        Args:
            tool_name: Name of the tool to run
            tool_args: Arguments to pass to the tool (default: empty string)

        Returns:
            Result from the tool execution
        """
        # Use dictionary lookup for O(1) performance
        target_tool = self._tool_dict.get(tool_name)

        if target_tool is None:
            available_tools = list(self._tool_dict.keys())
            raise ValueError(
                f"Unknown tool: {tool_name}. Available: {available_tools}")

        # Create proper context using the same pattern as the SDK
        ctx = RunContextWrapper(self.state)

        # Call the tool's invoke method directly with proper context
        result = await target_tool.on_invoke_tool(ctx, tool_args)

        # State is modified in-place through the context wrapper
        return result

    async def run_step(self, user_input: str) -> str:
        """Run a workflow step with persistent state"""
        result = await Runner.run(
            self,
            user_input,
            session=self.session,
            context=self.state,  # Use our managed state
            max_turns=50  # Increased for complete 9-step workflow
        )
        return result.final_output


async def main():
    """Main function to create agent and run complete workflow"""
    print("🚀 Creating NewsletterAgent...")

    # Load environment variables like the notebook does
    dotenv.load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Create agent with persistent state (timeout set in __init__)
    agent = NewsletterAgent(session_id="test_newsletter",
                            verbose=True, timeout=300.0)

    # User prompt to run complete workflow
    user_prompt = "Run all the workflow steps in order and create the newsletter"

    print(f"\n📝 User prompt: '{user_prompt}'")
    print("=" * 80)

    # Run the agent with persistent state
    start_time = time.time()
    result = await agent.run_step(user_prompt)
    duration = time.time() - start_time

    print("=" * 80)
    print(f"⏱️  Total execution time: {duration:.2f}s")
    print("📊 Final result:")
    print(result)


if __name__ == "__main__":
    print("🔧 NewsletterAgent - Complete Workflow Test")
    print("=" * 60)
    asyncio.run(main())
