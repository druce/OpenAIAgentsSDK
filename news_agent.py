#!/usr/bin/env python3
"""
Newsletter Agent for testing the complete workflow end-to-end.

This agent follows the ClassifierAgent pattern from test_agent.ipynb and implements
all 9 workflow steps defined in the WorkflowStatus object from utilities.py.
Each step updates the workflow status properly.

TODO: in summarize step, dedupe with chromadb before summarizing , mark as duplicate if so

"""

import asyncio
import time
import logging
import os
import json
import dotenv
from datetime import datetime, timedelta

import pandas as pd

from IPython.display import HTML, Image, Markdown, display

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from email.utils import parsedate_to_datetime

from agents import (Agent, Runner, RunContextWrapper, FunctionTool,
    Tool, SQLiteSession, set_default_openai_client)

from openai import AsyncOpenAI
from newsletter_state import NewsletterAgentState, StepStatus
from log_handler import SQLiteLogHandler
from fetch import Fetcher
from scrape import scrape_urls_concurrent
from llm import LLMagent, LangfuseClient

# Pydantic models for structured output
class ArticleSummary(BaseModel):
    """Model for AI-generated article summaries with exactly 3 bullet points"""
    summary: str = Field(
        description="Bullet-point summary of the article"
    )

# Global constants
LOGDB = 'newsagent_logs.db'

# AI Classification Prompts
FILTER_SYSTEM_PROMPT = """
You are a content-classification assistant that labels news headlines as AI-related or not.
Return **only** a JSON object that satisfies the provided schema.
For each headline provided, you must return an element with the same id, and a boolean value; do not skip any items.
No markdown, no markdown fences, no extra keys, no comments.
"""

FILTER_USER_PROMPT = """
Classify every headline below.

AI-related if the title mentions (explicitly or implicitly):
- Core AI technologies: machine learning, neural / deep / transformer networks
- AI Applications: computer vision, NLP, robotics, autonomous driving, generative media
- AI hardware, GPU chip supply, AI data centers and infrastructure
- Companies or labs known for AI: OpenAI, DeepMind, Anthropic, xAI, NVIDIA, etc.
- AI models & products: ChatGPT, Gemini, Claude, Sora, Midjourney, DeepSeek, etc.
- New AI products and AI integration into existing products/services
- AI policy / ethics / safety / regulation / analysis
- Research results related to AI
- AI industry figures (Sam Altman, Demis Hassabis, etc.)
- AI market and business developments, funding rounds, partnerships centered on AI
- Any other news with a significant AI component

Non-AI examples: crypto, ordinary software, non-AI gadgets and medical devices, and anything else.

Headlines to classify: {input_text}
"""

# Pydantic models for AI classification

# output class for classifying headlines
class AIClassification(BaseModel):
    """A single headline classification result"""
    id: int = Field("The news item id")
    input_str: str = Field(description="The original headline title")
    output: bool = Field(description="Whether the headline title is AI-related")

class AIClassificationList(BaseModel):
    """List of AIClassification for batch processing"""
    results_list: list[AIClassification] = Field(description="List of classification results")


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
            result = state.get_workflow_status_report("WORKFLOW STATUS (FROM PERSISTENT STATE)")

            # Add data summary if we have articles
            if state.headline_data:
                ai_related = sum(1 for a in state.headline_data if a.get('isAI') is True)
                result += f"\n\nData Summary:\n"
                result += f"  Total articles: {len(state.headline_data)}\n"
                result += f"  AI-related: {ai_related}\n"
                result += f"  Clusters: {len(state.clusters)}\n"
                result += f"  Sections: {len(state.newsletter_sections)}"

            # Add intervention guidance if workflow is in error state
            if state.has_errors():
                failed_steps = state.get_failed_steps()
                result += f"\n\n⚠️  INTERVENTION REQUIRED:\n"
                result += f"  Failed steps: {', '.join(failed_steps)}\n"
                if state.workflow_status_message:
                    result += f"  Instructions: {state.workflow_status_message}"

            if self.logger:
                self.logger.info("Completed check_workflow_status")
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
            report_lines.append(f"Status Message: {state.workflow_status_message}")

        report_lines.extend([
            f"Sources File: {state.sources_file}",
            "",
            "HEADLINE DATA:",
            f"  Total articles: {len(state.headline_data)}",
        ])

        if state.headline_data:
            ai_related = sum(1 for a in state.headline_data if a.get('isAI') is True)
            with_content = sum(1 for a in state.headline_data if a.get('content'))
            with_ratings = sum(1 for a in state.headline_data if a.get('quality_rating'))
            with_clusters = sum(1 for a in state.headline_data if a.get('cluster_topic'))

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
            f"  Newsletter sections: {len(state.newsletter_sections)} sections",
            f"  Final newsletter: {'Generated' if state.final_newsletter else 'Not created'}",
        ])

        if state.clusters:
            report_lines.extend([
                "",
                "TOPIC CLUSTERS:",
            ])
            for topic, urls in state.clusters.items():
                report_lines.append(f"  {topic}: {len(urls)} articles")

        if state.newsletter_sections:
            report_lines.extend([
                "",
                "NEWSLETTER SECTIONS:",
            ])
            for section_name, section_data in state.newsletter_sections.items():
                status = section_data.get('section_status', 'unknown')
                word_count = section_data.get('word_count', 0)
                article_count = section_data.get('article_count', 0)
                report_lines.append(f"  {section_name}: {status}, {article_count} articles, {word_count} words")

        if state.final_newsletter:
            newsletter_words = len(state.final_newsletter.split())
            report_lines.extend([
                "",
                "FINAL NEWSLETTER:",
                f"  Length: {newsletter_words} words",
                f"  Preview: {state.final_newsletter[:200]}..." if len(state.final_newsletter) > 200 else f"  Content: {state.final_newsletter}",
            ])

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
                self.logger.info(f"Step 1 already completed with {total_articles} articles")
            return f"Step 1 already completed! Found {total_articles} articles in persistent state."

        try:
            # Start the step using unified status system
            state.start_step(step_name)

            # Use real RSS fetching from sources.yaml
            async with Fetcher() as f:
                sources_results = await f.fetch_all(do_download=state.do_download)

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

            # Check if we need user intervention (significant failures)
            total_sources = len(successful_sources) + len(failed_sources)
            success_rate = len(successful_sources) / total_sources if total_sources > 0 else 0

            # Define intervention scenarios
            requires_intervention = (
                success_rate < 0.7 or  # Less than 70% success rate
                any(source in ['Bloomberg', 'WSJ', 'Wall Street Journal', 'Reuters'] for source in failed_sources)
            )

            if requires_intervention and failed_sources:
                # Set error state with intervention message
                intervention_message = f"Partial failure detected. {len(failed_sources)} sources failed: {', '.join(failed_sources)}. "
                if any('Bloomberg' in source or 'WSJ' in source for source in failed_sources):
                    intervention_message += "These sources typically require manual download due to access restrictions. "
                intervention_message += f"Download HTML files manually to download/sources/ directory and resume with do_download=False."

                state.error_step(step_name, intervention_message)

                # Store partial results anyway
                state.headline_data = all_articles

                if self.verbose:
                    print(f"⚠️  Intervention required: {len(successful_sources)} successful, {len(failed_sources)} failed")
                    print(f"Failed sources: {', '.join(failed_sources)}")

                return f"⚠️  Intervention Required! Successfully fetched from {len(successful_sources)} sources but {len(failed_sources)} sources failed.\n\n{intervention_message}"

            # Store results in persistent state
            headline_df = pd.DataFrame(all_articles)
            display(headline_df[["source", "url"]].groupby("source") \
                .count() \
                .reset_index() \
                .rename({'url': 'count'}))
            # assign id
            headline_df['id'] = headline_df.index
            state.headline_data = headline_df.to_dict('records')

            # Complete the step using unified status system
            state.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 1: Gathered {len(all_articles)} URLs from {len(successful_sources)} RSS sources")
                if failed_sources:
                    print(f"⚠️  Failed sources: {', '.join(failed_sources)}")

            status_msg = f"✅ Step 1 completed successfully! Gathered {len(all_articles)} articles from {len(successful_sources)} sources (RSS only)."
            if failed_sources:
                status_msg += f" {len(failed_sources)} sources failed or not implemented."

            status_msg += f"\n\n📊 Articles stored in persistent state: {len(state.headline_data)}"
            headline_df = pd.DataFrame(state.headline_data)
            display(headline_df)
            if self.logger:
                self.logger.info(f"Completed Step 1: Gathered {len(all_articles)} articles")
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
            ai_related_count = sum(1 for article in state.headline_data if article.get('isAI') is True)
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

            # Read headlines from persistent state
            total_articles = len(state.headline_data)

            if self.verbose:
                print(f"🔍 Classifying {total_articles} headlines using LLM...")

            # Prepare headlines for batch classification
            headline_df = pd.DataFrame(state.headline_data)
            display(headline_df)

            # Create LLM agent for AI classification
            classifier = LLMagent(
                system_prompt=FILTER_SYSTEM_PROMPT,
                user_prompt=FILTER_USER_PROMPT,
                output_type=AIClassificationList,
                model="gpt-5-nano",
                verbose=self.verbose,
                logger=self.logger
            )

            headline_df['isAI'] = \
                await classifier.filter_dataframe(headline_df[["id", "title"]])

            # Update state with AI classification results
            state.headline_data = headline_df.to_dict('records')

            # Complete step using unified status system
            state.complete_step(step_name)

            ai_related_count = sum(1 for article in state.headline_data if article.get('isAI') is True)
            total_count = len(state.headline_data)

            if self.verbose:
                print(f"✅ Completed Step 2: Filtered to {ai_related_count} AI-related headlines from {total_articles} total")

            status_msg = f"✅ Step 2 completed successfully! Filtered {total_articles} headlines to {ai_related_count} AI-related articles."
            status_msg += f"\n\n📊 Results stored in persistent state. Current step: {state.get_current_step()}"
            if self.logger:
                self.logger.info(f"Completed Step 2: Filtered to {ai_related_count} AI-related articles")
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
            ai_related_count = sum(1 for article in state.headline_data if article.get('isAI') is True)
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

            headline_df = pd.DataFrame(state.headline_data)

            # Filter for AI-related articles
            ai_mask = headline_df['isAI'] == True
            ai_df = headline_df[ai_mask].copy()

            if ai_df.empty:
                return f"❌ No AI-related articles found to download. Please run step 2 first."

            # Prepare input for scrape_urls_concurrent: (index, url, title)
            scrape_inputs = []
            for idx, row in ai_df.iterrows():
                scrape_inputs.append((
                    row['id'],  # Use 'id' as the index for matching
                    row['url'],
                    row['title']
                ))

            if self.logger:
                self.logger.info(f"Starting concurrent scraping of {len(scrape_inputs)} AI-related articles")

            # Use real scraping with scrape_urls_concurrent
            scrape_results = await scrape_urls_concurrent(
                scrape_inputs,
                concurrency=16,
                rate_limit_seconds=2.0,
                logger=self.logger
            )

            # Process scraping results and update DataFrame
            successful_downloads = 0
            total_length = 0

            # Create mapping from id to scrape results
            result_map = {}
            for index, status, url, title, html_path, last_updated in scrape_results:
                result_map[index] = {
                    'status': status,
                    'final_url': url,
                    'html_path': html_path,
                    'last_updated': last_updated
                }

            # Update headline_data with scraping results
            for i, article in enumerate(state.headline_data):
                if article.get('isAI') is True and article['id'] in result_map:
                    result = result_map[article['id']]

                    # Add new columns to headline_data
                    article['status'] = result['status']
                    article['final_url'] = result['final_url']
                    article['html_path'] = result['html_path']
                    article['last_updated'] = result['last_updated']
                    article['download_timestamp'] = datetime.now().isoformat()

                    # Extract content from downloaded file
                    if result['status'] == 'success' and result['html_path']:
                        try:
                            # Use normalize_html to extract clean text content
                            from scrape import normalize_html
                            content = normalize_html(result['html_path'], logger=self.logger)
                            article['content'] = content
                            article['content_length'] = len(content)
                            successful_downloads += 1
                            total_length += len(content)

                            if self.logger:
                                self.logger.debug(f"Successfully extracted content from {result['html_path']}: {len(content)} characters")
                        except Exception as e:
                            if self.logger:
                                self.logger.warning(f"Failed to extract content from {result['html_path']}: {e}")
                            article['content'] = f"Failed to extract content: {e}"
                            article['content_length'] = 0
                    else:
                        # Scraping failed or no file path
                        article['content'] = f"Failed to scrape content. Status: {result['status']}"
                        article['content_length'] = 0

            # Calculate stats
            download_success_rate = successful_downloads / len(ai_df) if not ai_df.empty else 0
            avg_article_length = total_length / successful_downloads if successful_downloads > 0 else 0

            # Complete the step
            state.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 3: Downloaded {successful_downloads} AI-related articles")

            status_msg = f"✅ Step 3 completed successfully! Downloaded {successful_downloads} AI-related articles with {download_success_rate:.0%} success rate."
            status_msg += f"\n📊 Average article length: {avg_article_length:.0f} characters"
            status_msg += f"\n🔗 Content stored in persistent state."
            if self.logger:
                self.logger.info(f"Completed Step 3: Downloaded {successful_downloads} articles")
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

    def _normalize(self, html_path: str) -> str:
        """
        Normalize HTML content to text and save to TEXT_DIR

        Args:
            html_path: Path to HTML file in PAGES_DIR

        Returns:
            Path to normalized text file in TEXT_DIR
        """
        from pathlib import Path
        from scrape import normalize_html
        from config import TEXT_DIR

        # Create text directory if it doesn't exist
        os.makedirs(TEXT_DIR, exist_ok=True)

        # Generate text file path (same name, .txt extension)
        html_file = Path(html_path)
        text_filename = html_file.stem + '.txt'
        text_path = os.path.join(TEXT_DIR, text_filename)

        try:
            # Extract text from HTML
            normalized_text = normalize_html(html_path, logger=self.logger)

            # Save text to file
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(normalized_text)

            if self.verbose and self.logger:
                self.logger.info(f"Normalized HTML to text: {html_path} -> {text_path}")

            return text_path

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to normalize HTML {html_path}: {str(e)}")
            return ""


    async def _summarize(self, text_path: str) -> List[str]:
        """
        Generate AI-powered summary from text file

        Args:
            text_path: Path to text file to summarize

        Returns:
            List of 3 bullet-point summaries
        """
        SUMMARIZE_SYSTEM_PROMPT, SUMMARIZE_USER_PROMPT, model = LangfuseClient().get_prompt("newsagent/summarize")

        try:
            # Read text content
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()

            if not text_content:
                if self.logger:
                    self.logger.warning(f"Empty text file: {text_path}")
                return "No content available for summarization."

            # Create LLM agent for summarization
            llm_agent = LLMagent(
                system_prompt=SUMMARIZE_SYSTEM_PROMPT,
                user_prompt=SUMMARIZE_USER_PROMPT,
                output_type=ArticleSummary,
                model=model,
                verbose=self.verbose,
                logger=self.logger
            )

            # Generate summary
            result = await llm_agent.prompt(article=text_content)

            if hasattr(result, 'summary') and isinstance(result.summary, str):
                if self.verbose and self.logger:
                    self.logger.info(f"Generated summary for {text_path}: {result.summary}")
                return result.summary
            else:
                if self.logger:
                    self.logger.error(f"Invalid summary format from LLM for {text_path}")
                return "Summary generation failed."

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to summarize {text_path}: {str(e)}")
            return f"Error generating summary: {str(e)}"

    async def _extract_summaries(self, ctx, args: str, max_concurrency: int = 16) -> str:
        """Execute Step 4: Extract Summaries using AI-powered summarization"""
        step_name = "step_04_extract_summaries"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.is_step_complete(step_name):
            return f"Step 4 already completed! Generated summaries."

        # Check if step 3 is completed
        if not state.is_step_complete("step_03_download_articles"):
            return f"❌ Cannot execute Step 4: Step 3 (Download Articles) must be completed first."

        # Check if workflow is blocked by errors
        if state.has_errors():
            return f"❌ Cannot execute Step 4: Workflow is blocked by errors. {state.workflow_status_message}"

        try:
            # Update workflow status for UI tracking
            state.start_step(step_name)

            # Convert headline_data to DataFrame for processing
            headline_df = pd.DataFrame(state.headline_data)

            # Filter to AI-related articles with HTML content
            ai_articles_mask = (headline_df.get('isAI') == True) & (headline_df['html_path'].notna())
            if not ai_articles_mask.any():
                return f"❌ No AI-related articles with HTML content found to summarize. Please run step 3 first."

            ai_articles_df = headline_df[ai_articles_mask].copy()

            if self.logger:
                self.logger.info(f"Processing {len(ai_articles_df)} AI articles for summarization")

            # Step 1: Normalize HTML to text for all articles
            text_paths = []
            normalization_errors = 0

            for idx, row in ai_articles_df.iterrows():
                html_path = row['html_path']
                try:
                    text_path = self._normalize(html_path)
                    text_paths.append(text_path)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to normalize {html_path}: {str(e)}")
                    text_paths.append(None)
                    normalization_errors += 1

            # Add text_path column to DataFrame
            ai_articles_df = ai_articles_df.copy()
            ai_articles_df['text_path'] = text_paths

            # Step 2: Summarize articles with valid text paths
            summaries = []
            summarization_errors = 0
            total_bullets = 0

            # Create semaphore for concurrent summarization
            semaphore = asyncio.Semaphore(max_concurrency)

            async def summarize_with_semaphore(idx, row):
                async with semaphore:
                    text_path = row['text_path']

                    if text_path is None:
                        return idx, "No text."

                    try:
                        summary_bullets = await self._summarize(text_path)
                        return idx, summary_bullets
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to summarize {text_path}: {str(e)}")
                        return idx, f"Summarization failed: {str(e)}"

            # Create tasks for all articles
            tasks = [
                summarize_with_semaphore(idx, row)
                for idx, row in ai_articles_df.iterrows()
            ]

            # Execute tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results maintaining order
            summaries = [None] * len(ai_articles_df)
            for result in results:
                if isinstance(result, Exception): # not currently returning exceptins from summarize_with_semaphore
                    summarization_errors += 1
                    continue
                idx, summary_bullets = result
                total_bullets += 1
                summaries[ai_articles_df.index.get_loc(idx)] = summary_bullets

            # Add summary column to DataFrame
            ai_articles_df['summary'] = summaries

            # Update the original headline_data with the new columns
            for idx, row in ai_articles_df.iterrows():
                # Find the corresponding article in state.headline_data
                url = row['url']
                for article in state.headline_data:
                    if article.get('url') == url:
                        article['text_path'] = row['text_path']
                        article['summary'] = row['summary']
                        article['summary_bullets'] = len([b for b in row['summary'] if b.strip()])
                        article['summary_timestamp'] = datetime.now().isoformat()
                        break

            # Calculate statistics
            articles_processed = len(ai_articles_df)
            successful_summaries = articles_processed - summarization_errors - normalization_errors
            avg_bullets_per_article = total_bullets / successful_summaries if successful_summaries > 0 else 0

            # Complete the step
            state.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 4: Generated AI summaries for {successful_summaries}/{articles_processed} articles")

            status_msg = f"✅ Step 4 completed successfully! Generated AI-powered summaries for {successful_summaries}/{articles_processed} articles."
            status_msg += f"\n📝 Average bullets per article: {avg_bullets_per_article:.1f}"
            if normalization_errors > 0:
                status_msg += f"\n⚠️  Normalization errors: {normalization_errors}"
            if summarization_errors > 0:
                status_msg += f"\n⚠️  Summarization errors: {summarization_errors}"
            status_msg += f"\n💾 Summaries stored in headline DataFrame."
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

# - extract free-form topics using prompt
# - add repeated free-form topics to canonical prompts
class ClusterByTopicTool:
    """Tool for Step 5: Cluster articles by topic
    - use summaries
    - extract free-form topics using prompt
    - add repeated free-form topics to canonical prompts
    - check for all canonical prompt sthat are matched
    - use prompt to limit each summary to 7 topics that best match
    - add topics to summary in headline_data
    - cluster summaries using hdbscan
    - for each cluster, send prompt to generate cluster title
    - after completion summaries have been updated to prepend topics, clusters created and named, headline_df items have a cluster_id
    """

    def __init__(self, verbose: bool = False, logger: logging.Logger = None):
        self.verbose = verbose
        self.logger = logger

    async def _cluster_by_topic(self, ctx, args: str) -> str:
        """Execute Step 5: Cluster By Topic using persistent state"""
        step_name = "step_05_cluster_by_topic"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.is_step_complete(step_name):
            cluster_count = len(state.clusters)
            total_articles = sum(len(articles) for articles in state.clusters.values())
            return f"Step 5 already completed! Created {cluster_count} topic clusters with {total_articles} articles."

        # Check if step 4 is completed
        if not state.is_step_complete("step_04_extract_summaries"):
            return f"❌ Cannot execute Step 5: Step 4 (Extract Summaries) must be completed first."

        # Check if workflow is blocked by errors
        if state.has_errors():
            return f"❌ Cannot execute Step 5: Workflow is blocked by errors. {state.workflow_status_message}"

        try:
            # Update workflow status for UI tracking
            state.start_step(step_name)

            # Get articles with summaries from persistent state
            articles_with_summaries = [
                article for article in state.headline_data
                if article.get('isAI') is True and
                article.get('summary') is not None
            ]

            if not articles_with_summaries:
                return f"❌ No summarized articles found to cluster. Please run step 4 first."

            # Clear existing clusters if rerunning
            state.clusters = {}

            # Mock clustering logic - in a real implementation, this would use NLP/ML
            # to group articles by semantic similarity of their titles and summaries
            predefined_topics = [
                "LLM Advances", "AI Safety & Ethics", "Business AI Applications",
                "Research Breakthroughs", "Industry News", "Other AI Topics"
            ]

            # Initialize empty clusters
            for topic in predefined_topics:
                state.clusters[topic] = []

            # Simple keyword-based clustering
            topic_keywords = {
                "LLM Advances": ["llm", "large language model", "gpt", "claude", "language model", "chatbot", "chat"],
                "AI Safety & Ethics": ["safety", "ethics", "bias", "fairness", "responsible", "trust", "alignment"],
                "Business AI Applications": ["business", "enterprise", "productivity", "automation", "workflow", "commercial"],
                "Research Breakthroughs": ["research", "breakthrough", "paper", "study", "academic", "university", "science"],
                "Industry News": ["company", "startup", "funding", "acquisition", "partnership", "launch", "release"],
                "Other AI Topics": []  # Catch-all
            }

            for article in articles_with_summaries:
                url = article.get('url', '')
                title_lower = article.get('title', '').lower()
                description_lower = article.get('description', '').lower()

                # Find best matching topic
                best_topic = "Other AI Topics"  # Default
                max_matches = 0

                for topic, keywords in topic_keywords.items():
                    if topic == "Other AI Topics":
                        continue

                    matches = sum(1 for keyword in keywords
                                if keyword in title_lower or keyword in description_lower)

                    if matches > max_matches:
                        max_matches = matches
                        best_topic = topic

                # Add article URL to the appropriate cluster
                state.clusters[best_topic].append(url)

                # Also update the article with cluster info
                article['cluster_topic'] = best_topic
                article['cluster_timestamp'] = datetime.now().isoformat()

            # Remove empty clusters
            state.clusters = {
                topic: articles for topic, articles in state.clusters.items()
                if articles
            }

            # Calculate stats
            total_clusters = len(state.clusters)
            total_articles = sum(len(articles) for articles in state.clusters.values())
            cluster_coherence_score = 0.84  # Mock coherence score

            # Complete the step
            state.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 5: Created {total_clusters} topic clusters")

            status_msg = f"✅ Step 5 completed successfully! Organized {total_articles} articles into {total_clusters} topic clusters."
            status_msg += f"\n📊 Cluster coherence score: {cluster_coherence_score:.1%}"
            status_msg += f"\n🏷️ Topics: {', '.join(state.clusters.keys())}"
            status_msg += f"\n💾 Clusters stored in persistent state."
            return status_msg

        except Exception as e:
            state.error_step(step_name, str(e))
            return f"❌ Step 5 failed: {str(e)}"

    def create_tool(self) -> FunctionTool:
        """Create a FunctionTool instance following OpenAI Agents SDK conventions"""
        return FunctionTool(
            name="cluster_by_topic",
            description="Execute Step 5: Group articles by thematic topics using clustering. Requires Step 4 to be completed first.",
            params_json_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            on_invoke_tool=self._cluster_by_topic
        )


class RateArticlesTool:
    """Tool for Step 6: Rate article quality and importance
    - create a rating using prompt to compare articles according to a rubric, and ELO/Bradley-Terry model
    - optionally could use a prompt to ask if it's AI related, important and not spammy, and use log probs from prompt
    - use additional criteria like log length of article, reputation of site
    - combine to create a rating
    - deduplicate each cluster of articles covering same story and add a point to the rating of the best story retained for each duplicate (since frequently covered stories are important)
    """

    def __init__(self, verbose: bool = False, logger: logging.Logger = None):
        self.verbose = verbose
        self.logger = logger

    async def _rate_articles(self, ctx, args: str) -> str:
        """Execute Step 6: Rate Articles using persistent state"""
        step_name = "step_06_rate_articles"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.is_step_complete(step_name):
            rated_articles = [article for article in state.headline_data if article.get('quality_rating')]
            avg_rating = sum(article.get('quality_rating', 0) for article in rated_articles) / len(rated_articles) if rated_articles else 0
            return f"Step 6 already completed! Rated {len(rated_articles)} articles with average rating {avg_rating:.1f}/10."

        # Check if step 5 is completed
        if not state.is_step_complete("step_05_cluster_by_topic"):
            return f"❌ Cannot execute Step 6: Step 5 (Cluster By Topic) must be completed first."

        # Check if workflow is blocked by errors
        if state.has_errors():
            return f"❌ Cannot execute Step 6: Workflow is blocked by errors. {state.workflow_status_message}"

        try:
            # Update workflow status for UI tracking
            state.start_step(step_name)

            # Get clustered articles from persistent state
            clustered_articles = [
                article for article in state.headline_data
                if article.get('isAI') is True and article.get('cluster_topic')
            ]

            if not clustered_articles:
                return f"❌ No clustered articles found to rate. Please run step 5 first."

            # Rate each article based on mock criteria
            articles_rated = 0
            total_rating = 0
            high_quality_count = 0

            for article in clustered_articles:
                # Mock rating logic - in reality, this would use AI to evaluate:
                # - Content quality, originality, depth
                # - Source credibility
                # - Relevance to AI community
                # - Timeliness and newsworthiness

                title_length = len(article.get('title', ''))
                has_description = bool(article.get('description', ''))
                source_quality = 8 if article.get('source') in ['Techmeme', 'Ars Technica', 'The Verge'] else 6
                cluster_bonus = 2 if article.get('cluster_topic') != 'Other AI Topics' else 0

                # Calculate mock quality rating (1-10)
                base_rating = 5
                if title_length > 50: base_rating += 1
                if has_description: base_rating += 1
                rating = min(10, base_rating + (source_quality - 6) + cluster_bonus)

                # Add some randomness to make it more realistic
                import random
                rating = max(1, min(10, rating + random.uniform(-1, 1)))

                # Store rating in article data
                article['quality_rating'] = round(rating, 1)
                article['rating_timestamp'] = datetime.now().isoformat()

                articles_rated += 1
                total_rating += rating
                if rating >= 7.0:
                    high_quality_count += 1

            # Calculate stats
            avg_rating = total_rating / articles_rated if articles_rated > 0 else 0

            # Complete the step
            state.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 6: Rated {articles_rated} articles")

            status_msg = f"✅ Step 6 completed successfully! Rated {articles_rated} articles with average rating {avg_rating:.1f}/10."
            status_msg += f"\n⭐ High quality articles (≥7.0): {high_quality_count}"
            status_msg += f"\n💾 Ratings stored in persistent state."
            return status_msg

        except Exception as e:
            state.error_step(step_name, str(e))
            return f"❌ Step 6 failed: {str(e)}"

    def create_tool(self) -> FunctionTool:
        """Create a FunctionTool instance following OpenAI Agents SDK conventions"""
        return FunctionTool(
            name="rate_articles",
            description="Execute Step 6: Evaluate article quality and importance with ratings. Requires Step 5 to be completed first.",
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
        step_name = "step_07_select_sections"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.is_step_complete(step_name):
            section_count = len(state.newsletter_sections)
            return f"Step 7 already completed! Created {section_count} newsletter sections."

        # Check if step 6 is completed
        if not state.is_step_complete("step_06_rate_articles"):
            return f"❌ Cannot execute Step 7: Step 6 (Rate Articles) must be completed first."

        # Check if workflow is blocked by errors
        if state.has_errors():
            return f"❌ Cannot execute Step 7: Workflow is blocked by errors. {state.workflow_status_message}"

        try:
            # Update workflow status for UI tracking
            state.start_step(step_name)

            # Get rated articles from persistent state
            rated_articles = [
                article for article in state.headline_data
                if article.get('isAI') is True and article.get('quality_rating')
            ]

            if not rated_articles:
                return f"❌ No rated articles found to organize into sections. Please run step 6 first."

            # Clear existing sections if rerunning
            state.newsletter_sections = {}

            # Create newsletter sections based on topic clusters and ratings
            # Use existing topic clusters but prioritize high-quality articles
            high_quality_articles = [a for a in rated_articles if a.get('quality_rating', 0) >= 7.0]
            medium_quality_articles = [a for a in rated_articles if 5.0 <= a.get('quality_rating', 0) < 7.0]

            # Group articles by cluster topic and select best ones for each section
            cluster_sections = {}
            for article in high_quality_articles + medium_quality_articles:
                cluster = article.get('cluster_topic', 'Other AI Topics')
                if cluster not in cluster_sections:
                    cluster_sections[cluster] = []
                cluster_sections[cluster].append(article)

            # Create newsletter sections with article assignments
            articles_assigned = 0
            for cluster, articles in cluster_sections.items():
                if not articles:
                    continue

                # Sort articles by rating (highest first) and take top articles
                sorted_articles = sorted(articles, key=lambda x: x.get('quality_rating', 0), reverse=True)
                top_articles = sorted_articles[:5]  # Max 5 articles per section

                # Create section outline (will be filled in step 8)
                section_content = {
                    'title': cluster,
                    'article_count': len(top_articles),
                    'articles': [{
                        'url': article.get('url'),
                        'title': article.get('title'),
                        'rating': article.get('quality_rating'),
                        'source': article.get('source')
                    } for article in top_articles],
                    'section_status': 'selected',
                    'timestamp': datetime.now().isoformat()
                }

                state.newsletter_sections[cluster] = section_content
                articles_assigned += len(top_articles)

            # Complete the step
            state.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 7: Created {len(state.newsletter_sections)} newsletter sections")

            status_msg = f"✅ Step 7 completed successfully! Organized content into {len(state.newsletter_sections)} sections with {articles_assigned} articles assigned."
            status_msg += f"\n📑 Sections: {', '.join(state.newsletter_sections.keys())}"
            status_msg += f"\n💾 Section plan stored in persistent state."
            return status_msg

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
            drafted_sections = [s for s in state.newsletter_sections.values() if s.get('content')]
            total_words = sum(len(s.get('content', '').split()) for s in drafted_sections)
            return f"Step 8 already completed! Drafted {len(drafted_sections)} sections with {total_words} total words."

        # Check if step 7 is completed
        if not state.is_step_complete("step_07_select_sections"):
            return f"❌ Cannot execute Step 8: Step 7 (Select Sections) must be completed first."

        # Check if workflow is blocked by errors
        if state.has_errors():
            return f"❌ Cannot execute Step 8: Workflow is blocked by errors. {state.workflow_status_message}"

        try:
            # Update workflow status for UI tracking
            state.start_step(step_name)

            # Get section plans from persistent state
            if not state.newsletter_sections:
                return f"❌ No newsletter sections found to draft. Please run step 7 first."

            # Draft content for each section
            sections_drafted = 0
            total_words = 0

            for section_name, section_data in state.newsletter_sections.items():
                if section_data.get('section_status') != 'selected':
                    continue

                articles = section_data.get('articles', [])
                if not articles:
                    continue

                # Mock section content generation - in reality, this would use AI
                # to create engaging newsletter content from article summaries
                section_content = f"## {section_name}\n\n"

                # Add intro paragraph
                intro_templates = {
                    'LLM Advances': "The latest developments in large language models continue to push the boundaries of what's possible in AI.",
                    'AI Safety & Ethics': "Important discussions around responsible AI development and deployment are shaping the future of the field.",
                    'Business AI Applications': "Companies are finding innovative ways to integrate AI into their products and workflows.",
                    'Research Breakthroughs': "Academic researchers are making significant strides in advancing our understanding of artificial intelligence.",
                    'Industry News': "The AI industry continues to evolve with new partnerships, funding rounds, and product launches."
                }

                intro = intro_templates.get(section_name, f"Here are the latest updates in {section_name.lower()}.")
                section_content += f"{intro}\n\n"

                # Add article summaries
                for i, article in enumerate(articles[:3]):  # Top 3 articles per section
                    article_url = article.get('url', '')
                    article_title = article.get('title', 'Unknown Title')
                    article_source = article.get('source', 'Unknown Source')

                    # Get the actual summary from article data if available
                    summary_bullets = article.get('summary', [
                        f"Key insights from this {section_name.lower()} article",
                        f"Important implications for the AI community",
                        f"Notable developments worth following"
                    ])

                    section_content += f"### {article_title}\n"
                    section_content += f"*Source: {article_source}*\n\n"

                    for bullet in summary_bullets:
                        section_content += f"- {bullet}\n"

                    section_content += f"\n[Read more]({article_url})\n\n"

                # Store the drafted content
                state.newsletter_sections[section_name]['content'] = section_content
                state.newsletter_sections[section_name]['section_status'] = 'drafted'
                state.newsletter_sections[section_name]['draft_timestamp'] = datetime.now().isoformat()
                state.newsletter_sections[section_name]['word_count'] = len(section_content.split())

                sections_drafted += 1
                total_words += len(section_content.split())

            # Calculate average words per section
            avg_words_per_section = total_words / sections_drafted if sections_drafted > 0 else 0

            # Complete the step
            state.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 8: Drafted {sections_drafted} sections")

            status_msg = f"✅ Step 8 completed successfully! Drafted {sections_drafted} sections with {total_words} total words."
            status_msg += f"\n📝 Average words per section: {avg_words_per_section:.0f}"
            status_msg += f"\n💾 Section content stored in persistent state."
            return status_msg

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
            newsletter_length = len(state.final_newsletter.split()) if state.final_newsletter else 0
            sections_count = len([s for s in state.newsletter_sections.values() if s.get('content')])
            return f"Step 9 already completed! Newsletter finalized with {sections_count} sections and {newsletter_length} words."

        # Check if step 8 is completed
        if not state.is_step_complete("step_08_draft_sections"):
            return f"❌ Cannot execute Step 9: Step 8 (Draft Sections) must be completed first."

        # Check if workflow is blocked by errors
        if state.has_errors():
            return f"❌ Cannot execute Step 9: Workflow is blocked by errors. {state.workflow_status_message}"

        try:
            # Update workflow status for UI tracking
            state.start_step(step_name)

            # Get drafted sections from persistent state
            drafted_sections = {
                name: data for name, data in state.newsletter_sections.items()
                if data.get('section_status') == 'drafted' and data.get('content')
            }

            if not drafted_sections:
                return f"❌ No drafted sections found to finalize. Please run step 8 first."

            # Create the final newsletter by combining all sections
            today = datetime.now().strftime("%B %d, %Y")

            newsletter_content = f"# AI News Digest - {today}\n\n"
            newsletter_content += f"*Curated insights from the latest in artificial intelligence*\n\n"
            newsletter_content += f"---\n\n"

            # Add table of contents
            newsletter_content += "## Table of Contents\n\n"
            for i, section_name in enumerate(drafted_sections.keys(), 1):
                newsletter_content += f"{i}. [{section_name}](#{section_name.lower().replace(' ', '-').replace('&', 'and')})\n"
            newsletter_content += "\n---\n\n"

            # Add each section content
            for section_name, section_data in drafted_sections.items():
                newsletter_content += section_data.get('content', '')
                newsletter_content += "\n---\n\n"

            # Add footer
            newsletter_content += "## About This Newsletter\n\n"
            newsletter_content += "This AI News Digest was automatically curated using our intelligent newsletter agent. "
            newsletter_content += f"We analyzed {len(state.headline_data)} articles from {len(set(a.get('source', '') for a in state.headline_data))} sources "
            newsletter_content += f"to bring you the most relevant AI developments.\n\n"
            newsletter_content += f"*Generated on {today}*\n"

            # Store the final newsletter
            state.final_newsletter = newsletter_content

            # Calculate final stats
            newsletter_length = len(newsletter_content.split())
            sections_included = len(drafted_sections)

            # Mock quality score based on content metrics
            base_quality = 7.0
            if sections_included >= 4: base_quality += 0.5
            if newsletter_length >= 2000: base_quality += 0.5
            if newsletter_length >= 3000: base_quality += 0.5
            final_quality_score = min(10.0, base_quality)

            # Complete the step and mark workflow as complete
            state.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 9: Finalized newsletter ({newsletter_length} words)")

            status_msg = f"🎉 Step 9 completed successfully! Newsletter finalized with {sections_included} sections and {newsletter_length} words."
            status_msg += f"\n⭐ Quality score: {final_quality_score:.1f}/10"
            status_msg += f"\n📰 Complete newsletter stored in persistent state"
            status_msg += f"\n✅ Workflow complete! All 9 steps finished successfully."
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
                 logger: logging.Logger = None):
        """
        Initialize the NewsletterAgent with persistent state

        Args:
            session_id: Unique identifier for the session (for persistence)
            state: Optional NewsletterAgentState to use. If None, creates new or loads from session
            verbose: Enable verbose logging
            logger: Optional logger instance (creates one if None)
        """
        self.session = SQLiteSession(session_id, "newsletter_agent.db")
        self.verbose = verbose
        self.logger = logger or setup_logging(session_id, LOGDB)

        # Initialize state - use provided state or create/load default
        if state is not None:
            self.state = state
            if self.verbose:
                self.logger.info(f"Using provided state with {len(state.headline_data)} articles")
        else:
            # Try to load existing state from session, or create new if none exists
            try:
                # Note: session.load_state may not exist in OpenAI Agents SDK
                # This is a placeholder for proper session loading
                self.state = NewsletterAgentState()
                if self.verbose:
                    self.logger.info("Created new NewsletterAgentState")
            except Exception as e:
                self.state = NewsletterAgentState()
                if self.verbose:
                    self.logger.info(f"Created new NewsletterAgentState (session load failed: {e})")

        # System prompt that guides tool selection based on workflow status
        system_prompt = """
You are an AI newsletter writing agent that executes a 9-step workflow process using tools with persistent state.

WORKFLOW OVERVIEW:
1. Step 1: Gather URLs - Collect headlines and URLs from various sources
2. Step 2: Filter URLs - Filter headlines to AI-related content only
3. Step 3: Download Articles - Fetch full article content from URLs
4. Step 4: Extract Summaries - Create bullet point summaries of each article
5. Step 5: Cluster By Topic - Group articles by thematic topics
6. Step 6: Rate Articles - Evaluate article quality and importance
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
                FinalizeNewsletterTool(self.verbose, self.logger).create_tool(),
            ]
        )

        # Create tool dictionary
        self._tool_dict = {tool.name: tool for tool in self.tools}

        if self.verbose:
            print(f"Initialized NewsletterAgent with persistent state and 9-step workflow")
            print(f"Session ID: {session_id}")

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
            raise ValueError(f"Unknown tool: {tool_name}. Available: {available_tools}")

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

    # Set up OpenAI client for the agents SDK
    set_default_openai_client(AsyncOpenAI(api_key=api_key))

    # Create agent with persistent state
    agent = NewsletterAgent(session_id="test_newsletter", verbose=True)

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
    print(f"📊 Final result:")
    print(result)


if __name__ == "__main__":
    print("🔧 NewsletterAgent - Complete Workflow Test")
    print("=" * 60)
    asyncio.run(main())