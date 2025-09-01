#!/usr/bin/env python3
"""
Newsletter Agent for testing the complete workflow end-to-end.

This agent follows the ClassifierAgent pattern from test_agent.ipynb and implements
all 9 workflow steps defined in the WorkflowStatus object from utilities.py.
Each step updates the workflow status properly.
"""

import asyncio
import time
import logging
import os
import json
import dotenv
import yaml
import feedparser
import aiohttp
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from agents import Agent, Runner, set_default_openai_client, FunctionTool, Tool, SQLiteSession
from openai import AsyncOpenAI
from utilities import WorkflowStatus, StepStatus, get_workflow_status_report, print_workflow_summary

class NewsletterAgentState(BaseModel):
    """Persistent state for the newsletter agent workflow"""

    # Serializable data storage (DataFrame as list of dicts)
    headline_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of headline dictionaries with columns: title, url, source, timestamp, ai_related, etc."
    )

    # Source management
    sources: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of source configurations loaded from YAML"
    )
    sources_file: str = Field(
        default="sources.yaml",
        description="YAML filename containing source configurations"
    )

    # Workflow progress
    current_step: int = Field(default=0, description="Current workflow step (0-9)")
    workflow_complete: bool = Field(default=False, description="Whether the entire workflow is complete")

    # Processing results
    article_summaries: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="URL -> list of bullet point summaries"
    )
    topic_clusters: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Topic name -> list of article URLs"
    )
    newsletter_sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Section name -> section content"
    )
    final_newsletter: str = Field(default="", description="Final newsletter content")

    # Configuration
    cluster_topics: List[str] = Field(
        default_factory=list,
        description="List of topic names for categorization"
    )
    max_edits: int = Field(default=3, description="Maximum editing iterations")
    n_browsers: int = Field(default=3, description="Number of concurrent browsers")

    # Helper methods for DataFrame conversion
    @property
    def headline_df(self) -> 'pd.DataFrame':
        """Convert stored data back to DataFrame"""
        import pandas as pd
        return pd.DataFrame(self.headline_data)

    def update_headlines(self, df: 'pd.DataFrame'):
        """Update headline data from DataFrame"""
        self.headline_data = df.to_dict('records')


async def fetch_rss(session: aiohttp.ClientSession, source_key: str, source_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch and parse RSS feed from a source record

    Args:
        session: aiohttp session for HTTP requests
        source_key: Top-level key from sources.yaml (e.g., "Techmeme")
        source_record: Full source record from sources.yaml

    Returns:
        Dict with source_key, results, status, metadata
    """
    rss_url = source_record.get('rss')
    if not rss_url:
        return {
            'source_key': source_key,
            'results': [],
            'status': 'error',
            'metadata': {'error': 'No RSS URL found in source record'}
        }

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with session.get(rss_url, timeout=timeout) as response:
            if response.status == 200:
                content = await response.text()
                feed = feedparser.parse(content)

                # Extract articles from feed entries
                articles = []
                for entry in feed.entries[:50]:  # Limit to 50 entries
                    article = {
                        'title': entry.get('title', ''),
                        'url': entry.get('link', ''),
                        'description': entry.get('description', '') or entry.get('summary', ''),
                        'published': entry.get('published', ''),
                        'source': source_key
                    }
                    articles.append(article)

                return {
                    'source_key': source_key,
                    'results': articles,
                    'status': 'success',
                    'metadata': {
                        'feed_title': feed.feed.get('title', ''),
                        'feed_description': feed.feed.get('description', ''),
                        'entries_count': len(articles),
                        'rss_url': rss_url
                    }
                }
            else:
                return {
                    'source_key': source_key,
                    'results': [],
                    'status': 'error',
                    'metadata': {'error': f'HTTP {response.status} from {rss_url}'}
                }

    except Exception as e:
        return {
            'source_key': source_key,
            'results': [],
            'status': 'error',
            'metadata': {'error': f'Failed to fetch RSS: {str(e)}'}
        }


async def fetch_html(session: aiohttp.ClientSession, source_key: str, source_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dummy function for HTML fetching - returns None for now

    Args:
        session: aiohttp session (unused)
        source_key: Top-level key from sources.yaml
        source_record: Full source record from sources.yaml

    Returns:
        Dict with source_key and None results
    """
    return {
        'source_key': source_key,
        'results': None,
        'status': 'not_implemented',
        'metadata': {'message': 'HTML fetching not yet implemented'}
    }


def fn_extract_newsapi(state: NewsletterAgentState) -> NewsletterAgentState:
    """
    Get AI news via newsapi - this is a placeholder function
    https://newsapi.org/docs/get-started
    """
    # This function is not currently integrated with the workflow
    # It's here for reference from the original implementation
    print("NewsAPI function called but not implemented in current workflow")
    return state


async def fetch_api(session: aiohttp.ClientSession, source_key: str, source_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dummy function for API fetching - returns None for now

    Args:
        session: aiohttp session (unused)
        source_key: Top-level key from sources.yaml
        source_record: Full source record from sources.yaml

    Returns:
        Dict with source_key and None results
    """

    return {
        'source_key': source_key,
        'results': None,
        'status': 'not_implemented',
        'metadata': {'message': 'API fetching not yet implemented'}
    }


async def gather_urls(sources_file: str = "sources.yaml", max_concurrent: int = 8) -> List[Dict[str, Any]]:
    """
    Load sources.yaml and fetch content from all sources concurrently

    Args:
        sources_file: Path to sources.yaml file
        max_concurrent: Maximum concurrent requests

    Returns:
        List of results from all sources
    """
    # Load sources from YAML file
    try:
        with open(sources_file, 'r', encoding='utf-8') as file:
            sources = yaml.safe_load(file) or {}
    except FileNotFoundError:
        return [{
            'source_key': 'error',
            'results': [],
            'status': 'error',
            'metadata': {'error': f'Sources file not found: {sources_file}'}
        }]
    except yaml.YAMLError as e:
        return [{
            'source_key': 'error',
            'results': [],
            'status': 'error',
            'metadata': {'error': f'Error parsing YAML: {str(e)}'}
        }]

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_semaphore(session, source_key, source_record):
        async with semaphore:
            # Determine which fetch function to use based on priority:
            # 1. RSS if available (highest priority)
            # 2. API if type is 'rest'
            # 3. HTML otherwise

            if source_record.get('rss'):
                return await fetch_rss(session, source_key, source_record)
            elif source_record.get('type') == 'rest':
                return await fetch_api(session, source_key, source_record)
            else:
                return await fetch_html(session, source_key, source_record)

    # Create tasks for all sources
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_with_semaphore(session, source_key, source_record)
            for source_key, source_record in sources.items()
            if isinstance(source_record, dict)  # Skip any malformed entries
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, dict):
                valid_results.append(result)
            else:
                # Log exception as error result
                valid_results.append({
                    'source_key': 'unknown',
                    'results': [],
                    'status': 'error',
                    'metadata': {'error': f'Exception during fetch: {str(result)}'}
                })

        return valid_results


class WorkflowStatusTool:
    """Tool to check current workflow status"""

    def __init__(self, workflow_status: WorkflowStatus):
        self.workflow_status = workflow_status

    async def _check_workflow_status(self, ctx, args: str) -> str:
        """Get current workflow status report based on persistent state"""
        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Create a status report based on persistent state
        step_names = [
            "step_01_gather_urls", "step_02_filter_urls", "step_03_download_articles",
            "step_04_extract_summaries", "step_05_cluster_by_topic", "step_06_rate_articles",
            "step_07_select_sections", "step_08_draft_sections", "step_09_finalize_newsletter"
        ]

        lines = [
            "WORKFLOW STATUS (FROM PERSISTENT STATE)",
            f"Current Step: {state.current_step}/9",
            f"Workflow Complete: {state.workflow_complete}",
            f"Progress: {(state.current_step/9)*100:.1f}%",
            "",
            "Step Details:"
        ]

        for i, step_name in enumerate(step_names, 1):
            if i <= state.current_step:
                status = "✅ completed"
            elif i == state.current_step + 1:
                status = "➡️ next to execute"
            else:
                status = "⭕ not started"

            formatted_name = step_name.replace('step_', 'Step ').replace('_', ' ').title()
            formatted_name = formatted_name.replace('0', '').replace('  ', ' ')  # Clean up numbering
            lines.append(f"  {formatted_name}: {status}")

        if state.headline_data:
            lines.extend([
                "",
                "Data Summary:",
                f"  Total articles: {len(state.headline_data)}",
                f"  AI-related: {sum(1 for a in state.headline_data if a.get('ai_related') is True)}",
                f"  Summaries: {len(state.article_summaries)}",
                f"  Clusters: {len(state.topic_clusters)}",
                f"  Sections: {len(state.newsletter_sections)}",
            ])

        return "\n".join(lines)

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

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    async def _inspect_state(self, ctx, args: str) -> str:
        """Inspect detailed state data for debugging and monitoring"""
        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Create detailed state report
        report_lines = [
            "DETAILED STATE INSPECTION",
            "=" * 50,
            f"Current Step: {state.current_step}/9",
            f"Workflow Complete: {state.workflow_complete}",
            f"Sources File: {state.sources_file}",
            "",
            "HEADLINE DATA:",
            f"  Total articles: {len(state.headline_data)}",
        ]

        if state.headline_data:
            ai_related = sum(1 for a in state.headline_data if a.get('ai_related') is True)
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
            f"  Article summaries: {len(state.article_summaries)} articles",
            f"  Topic clusters: {len(state.topic_clusters)} topics",
            f"  Newsletter sections: {len(state.newsletter_sections)} sections",
            f"  Final newsletter: {'Generated' if state.final_newsletter else 'Not created'}",
        ])

        if state.topic_clusters:
            report_lines.extend([
                "",
                "TOPIC CLUSTERS:",
            ])
            for topic, urls in state.topic_clusters.items():
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
    """Tool for Step 1: Gather URLs from various news sources"""

    def __init__(self, workflow_status: WorkflowStatus, verbose: bool = False):
        self.workflow_status = workflow_status  # Keep for UI progress tracking
        self.verbose = verbose

    async def _gather_urls(self, ctx, args: str) -> str:
        """Execute Step 1: Gather URLs using persistent state"""
        step_name = "step_01_gather_urls"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.current_step >= 1:
            total_articles = len(state.headline_data)
            return f"Step 1 already completed! Found {total_articles} articles in persistent state."

        try:
            # Update workflow status for UI tracking
            self.workflow_status.start_step(step_name)

            # Use real RSS fetching from sources.yaml
            sources_results = await gather_urls(state.sources_file, max_concurrent=5)

            # Process results and store in persistent state
            all_articles = []
            successful_sources = []
            failed_sources = []

            for result in sources_results:
                if result['status'] == 'success' and result['results']:
                    # Add source info to each article
                    for article in result['results']:
                        article['source_key'] = result['source_key']
                        article['ai_related'] = None  # To be determined in step 2
                        all_articles.append(article)
                    successful_sources.append(result['source_key'])
                elif result['status'] == 'not_implemented':
                    # Skip HTML/API sources for now
                    continue
                else:
                    failed_sources.append(result['source_key'])

            # Store results in persistent state
            state.headline_data = all_articles
            state.current_step = 1

            # Also update workflow status for UI
            self.workflow_status.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 1: Gathered {len(all_articles)} URLs from {len(successful_sources)} RSS sources")
                if failed_sources:
                    print(f"⚠️  Failed sources: {', '.join(failed_sources)}")

            status_msg = f"✅ Step 1 completed successfully! Gathered {len(all_articles)} articles from {len(successful_sources)} sources (RSS only)."
            if failed_sources:
                status_msg += f" {len(failed_sources)} sources failed or not implemented."

            status_msg += f"\n\n📊 Articles stored in persistent state: {len(state.headline_data)}"
            return status_msg

        except Exception as e:
            self.workflow_status.error_step(step_name, str(e))
            return f"❌ Step 1 failed: {str(e)}"

    def create_tool(self) -> FunctionTool:
        """Create a FunctionTool instance following OpenAI Agents SDK conventions"""
        return FunctionTool(
            name="gather_urls",
            description="Execute Step 1: Gather URLs and headlines from various news sources. Only use this tool if Step 1 is not already completed.",
            params_json_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            on_invoke_tool=self._gather_urls
        )


class FilterUrlsTool:
    """Tool for Step 2: Filter URLs to AI-related content"""

    def __init__(self, workflow_status: WorkflowStatus, verbose: bool = False):
        self.workflow_status = workflow_status
        self.verbose = verbose

    async def _filter_urls(self, ctx, args: str) -> str:
        """Execute Step 2: Filter URLs using persistent state"""
        step_name = "step_02_filter_urls"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.current_step >= 2:
            ai_related_count = sum(1 for article in state.headline_data if article.get('ai_related') is True)
            total_count = len(state.headline_data)
            return f"Step 2 already completed! Filtered {total_count} articles, {ai_related_count} identified as AI-related."

        # Check if step 1 is completed
        if state.current_step < 1 or not state.headline_data:
            return f"❌ Cannot execute Step 2: Step 1 (Gather URLs) must be completed first. Current step: {state.current_step}"

        try:
            # Update workflow status for UI tracking
            self.workflow_status.start_step(step_name)

            # Read headlines from persistent state
            total_articles = len(state.headline_data)

            # Mock AI classification - in a real implementation, this would use an AI model
            # to analyze titles and descriptions for AI relevance
            ai_related_count = 0
            for i, article in enumerate(state.headline_data):
                # Simple keyword-based mock classification
                title_lower = article.get('title', '').lower()
                description_lower = article.get('description', '').lower()

                ai_keywords = [
                    'artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning',
                    'neural network', 'llm', 'large language model', 'gpt', 'claude',
                    'openai', 'anthropic', 'chatbot', 'automation', 'algorithm',
                    'computer vision', 'natural language', 'nlp', 'robotics'
                ]

                is_ai_related = any(keyword in title_lower or keyword in description_lower
                                  for keyword in ai_keywords)

                # Update article with AI classification
                state.headline_data[i]['ai_related'] = is_ai_related
                if is_ai_related:
                    ai_related_count += 1

            # Update persistent state
            state.current_step = 2

            # Also update workflow status for UI
            self.workflow_status.complete_step(step_name)

            filter_accuracy = ai_related_count / total_articles if total_articles > 0 else 0

            if self.verbose:
                print(f"✅ Completed Step 2: Filtered to {ai_related_count} AI-related headlines from {total_articles} total")

            status_msg = f"✅ Step 2 completed successfully! Filtered {total_articles} headlines to {ai_related_count} AI-related articles (accuracy: {filter_accuracy:.1%})."
            status_msg += f"\n\n📊 Results stored in persistent state. Current step: {state.current_step}"
            return status_msg

        except Exception as e:
            self.workflow_status.error_step(step_name, str(e))
            return f"❌ Step 2 failed: {str(e)}"

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
    """Tool for Step 3: Download article content"""

    def __init__(self, workflow_status: WorkflowStatus, verbose: bool = False):
        self.workflow_status = workflow_status
        self.verbose = verbose

    async def _download_articles(self, ctx, args: str) -> str:
        """Execute Step 3: Download Articles using persistent state"""
        step_name = "step_03_download_articles"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.current_step >= 3:
            ai_articles = [article for article in state.headline_data if article.get('ai_related') is True]
            downloaded_count = sum(1 for article in ai_articles if article.get('content'))
            return f"Step 3 already completed! Downloaded content for {downloaded_count} AI-related articles."

        # Check if step 2 is completed
        if state.current_step < 2:
            return f"❌ Cannot execute Step 3: Step 2 (Filter URLs) must be completed first. Current step: {state.current_step}"

        try:
            # Update workflow status for UI tracking
            self.workflow_status.start_step(step_name)

            # Get AI-related articles from persistent state
            ai_articles = [article for article in state.headline_data if article.get('ai_related') is True]

            if not ai_articles:
                return f"❌ No AI-related articles found to download. Please run step 2 first."

            # Mock content download - in a real implementation, this would fetch actual article content
            successful_downloads = 0
            total_length = 0

            for article in state.headline_data:
                if article.get('ai_related') is True:
                    # Simulate downloading article content
                    # In reality, this would use web scraping or API calls
                    mock_content = f"Mock article content for: {article.get('title', 'Unknown title')}\n\n"
                    mock_content += f"This is placeholder content that would normally be extracted from {article.get('url', 'unknown URL')}.\n"
                    mock_content += f"The article covers topics related to AI and technology as indicated by the title and description.\n"
                    mock_content += f"Source: {article.get('source', 'Unknown source')}\n"

                    # Add content to the article data
                    article['content'] = mock_content
                    article['download_timestamp'] = datetime.now().isoformat()
                    article['content_length'] = len(mock_content)

                    successful_downloads += 1
                    total_length += len(mock_content)

            # Calculate stats
            download_success_rate = successful_downloads / len(ai_articles) if ai_articles else 0
            avg_article_length = total_length / successful_downloads if successful_downloads > 0 else 0

            # Update persistent state
            state.current_step = 3

            # Also update workflow status for UI
            self.workflow_status.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 3: Downloaded {successful_downloads} AI-related articles")

            status_msg = f"✅ Step 3 completed successfully! Downloaded {successful_downloads} AI-related articles with {download_success_rate:.0%} success rate."
            status_msg += f"\n📊 Average article length: {avg_article_length:.0f} characters"
            status_msg += f"\n🔗 Content stored in persistent state. Current step: {state.current_step}"
            return status_msg

        except Exception as e:
            self.workflow_status.error_step(step_name, str(e))
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
    """Tool for Step 4: Extract article summaries"""

    def __init__(self, workflow_status: WorkflowStatus, verbose: bool = False):
        self.workflow_status = workflow_status
        self.verbose = verbose

    async def _extract_summaries(self, ctx, args: str) -> str:
        """Execute Step 4: Extract Summaries using persistent state"""
        step_name = "step_04_extract_summaries"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.current_step >= 4:
            summary_count = len([url for url in state.article_summaries.keys() if state.article_summaries[url]])
            return f"Step 4 already completed! Generated summaries for {summary_count} articles."

        # Check if step 3 is completed
        if state.current_step < 3:
            return f"❌ Cannot execute Step 4: Step 3 (Download Articles) must be completed first. Current step: {state.current_step}"

        try:
            # Update workflow status for UI tracking
            self.workflow_status.start_step(step_name)

            # Get articles with content from persistent state
            articles_with_content = [
                article for article in state.headline_data
                if article.get('ai_related') is True and article.get('content')
            ]

            if not articles_with_content:
                return f"❌ No downloaded AI-related articles found to summarize. Please run step 3 first."

            # Clear existing summaries if rerunning
            state.article_summaries = {}

            # Generate summaries for each article
            articles_summarized = 0
            total_bullets = 0

            for article in articles_with_content:
                url = article.get('url', f"article_{articles_summarized}")
                title = article.get('title', 'Unknown title')
                content = article.get('content', '')

                # Mock summary generation - in a real implementation, this would use an AI model
                # to create bullet point summaries from the full article content
                mock_summary = [
                    f"Key insight from '{title[:50]}...' - Main technological development discussed",
                    f"Business implications or market impact highlighted in the article",
                    f"Future outlook or expert predictions mentioned in the content"
                ]

                # Store summary in persistent state
                state.article_summaries[url] = mock_summary
                articles_summarized += 1
                total_bullets += len(mock_summary)

                # Add summary reference to article data as well
                article['summary_bullets'] = len(mock_summary)
                article['summary_timestamp'] = datetime.now().isoformat()

            # Calculate stats
            avg_bullets_per_article = total_bullets / articles_summarized if articles_summarized > 0 else 0
            summary_quality_score = 0.89  # Mock quality score

            # Update persistent state
            state.current_step = 4

            # Also update workflow status for UI
            self.workflow_status.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 4: Created summaries for {articles_summarized} articles")

            status_msg = f"✅ Step 4 completed successfully! Generated {avg_bullets_per_article:.1f}-bullet summaries for {articles_summarized} articles."
            status_msg += f"\n📝 Quality score: {summary_quality_score:.1%}"
            status_msg += f"\n💾 Summaries stored in persistent state. Current step: {state.current_step}"
            return status_msg

        except Exception as e:
            self.workflow_status.error_step(step_name, str(e))
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


class ClusterByTopicTool:
    """Tool for Step 5: Cluster articles by topic"""

    def __init__(self, workflow_status: WorkflowStatus, verbose: bool = False):
        self.workflow_status = workflow_status
        self.verbose = verbose

    async def _cluster_by_topic(self, ctx, args: str) -> str:
        """Execute Step 5: Cluster By Topic using persistent state"""
        step_name = "step_05_cluster_by_topic"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.current_step >= 5:
            cluster_count = len(state.topic_clusters)
            total_articles = sum(len(articles) for articles in state.topic_clusters.values())
            return f"Step 5 already completed! Created {cluster_count} topic clusters with {total_articles} articles."

        # Check if step 4 is completed
        if state.current_step < 4:
            return f"❌ Cannot execute Step 5: Step 4 (Extract Summaries) must be completed first. Current step: {state.current_step}"

        try:
            # Update workflow status for UI tracking
            self.workflow_status.start_step(step_name)

            # Get articles with summaries from persistent state
            articles_with_summaries = [
                article for article in state.headline_data
                if article.get('ai_related') is True and
                article.get('url') in state.article_summaries
            ]

            if not articles_with_summaries:
                return f"❌ No summarized articles found to cluster. Please run step 4 first."

            # Clear existing clusters if rerunning
            state.topic_clusters = {}

            # Mock clustering logic - in a real implementation, this would use NLP/ML
            # to group articles by semantic similarity of their titles and summaries
            predefined_topics = [
                "LLM Advances", "AI Safety & Ethics", "Business AI Applications",
                "Research Breakthroughs", "Industry News", "Other AI Topics"
            ]

            # Initialize empty clusters
            for topic in predefined_topics:
                state.topic_clusters[topic] = []

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
                state.topic_clusters[best_topic].append(url)

                # Also update the article with cluster info
                article['cluster_topic'] = best_topic
                article['cluster_timestamp'] = datetime.now().isoformat()

            # Remove empty clusters
            state.topic_clusters = {
                topic: articles for topic, articles in state.topic_clusters.items()
                if articles
            }

            # Calculate stats
            total_clusters = len(state.topic_clusters)
            total_articles = sum(len(articles) for articles in state.topic_clusters.values())
            cluster_coherence_score = 0.84  # Mock coherence score

            # Update persistent state
            state.current_step = 5

            # Also update workflow status for UI
            self.workflow_status.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 5: Created {total_clusters} topic clusters")

            status_msg = f"✅ Step 5 completed successfully! Organized {total_articles} articles into {total_clusters} topic clusters."
            status_msg += f"\n📊 Cluster coherence score: {cluster_coherence_score:.1%}"
            status_msg += f"\n🏷️ Topics: {', '.join(state.topic_clusters.keys())}"
            status_msg += f"\n💾 Clusters stored in persistent state. Current step: {state.current_step}"
            return status_msg

        except Exception as e:
            self.workflow_status.error_step(step_name, str(e))
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
    """Tool for Step 6: Rate article quality and importance"""

    def __init__(self, workflow_status: WorkflowStatus, verbose: bool = False):
        self.workflow_status = workflow_status
        self.verbose = verbose

    async def _rate_articles(self, ctx, args: str) -> str:
        """Execute Step 6: Rate Articles using persistent state"""
        step_name = "step_06_rate_articles"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.current_step >= 6:
            rated_articles = [article for article in state.headline_data if article.get('quality_rating')]
            avg_rating = sum(article.get('quality_rating', 0) for article in rated_articles) / len(rated_articles) if rated_articles else 0
            return f"Step 6 already completed! Rated {len(rated_articles)} articles with average rating {avg_rating:.1f}/10."

        # Check if step 5 is completed
        if state.current_step < 5:
            return f"❌ Cannot execute Step 6: Step 5 (Cluster By Topic) must be completed first. Current step: {state.current_step}"

        try:
            # Update workflow status for UI tracking
            self.workflow_status.start_step(step_name)

            # Get clustered articles from persistent state
            clustered_articles = [
                article for article in state.headline_data
                if article.get('ai_related') is True and article.get('cluster_topic')
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

            # Update persistent state
            state.current_step = 6

            # Also update workflow status for UI
            self.workflow_status.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 6: Rated {articles_rated} articles")

            status_msg = f"✅ Step 6 completed successfully! Rated {articles_rated} articles with average rating {avg_rating:.1f}/10."
            status_msg += f"\n⭐ High quality articles (≥7.0): {high_quality_count}"
            status_msg += f"\n💾 Ratings stored in persistent state. Current step: {state.current_step}"
            return status_msg

        except Exception as e:
            self.workflow_status.error_step(step_name, str(e))
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


class SelectSectionsTool:
    """Tool for Step 7: Select newsletter sections"""

    def __init__(self, workflow_status: WorkflowStatus, verbose: bool = False):
        self.workflow_status = workflow_status
        self.verbose = verbose

    async def _select_sections(self, ctx, args: str) -> str:
        """Execute Step 7: Select Sections using persistent state"""
        step_name = "step_07_select_sections"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.current_step >= 7:
            section_count = len(state.newsletter_sections)
            return f"Step 7 already completed! Created {section_count} newsletter sections."

        # Check if step 6 is completed
        if state.current_step < 6:
            return f"❌ Cannot execute Step 7: Step 6 (Rate Articles) must be completed first. Current step: {state.current_step}"

        try:
            # Update workflow status for UI tracking
            self.workflow_status.start_step(step_name)

            # Get rated articles from persistent state
            rated_articles = [
                article for article in state.headline_data
                if article.get('ai_related') is True and article.get('quality_rating')
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

            # Update persistent state
            state.current_step = 7

            # Also update workflow status for UI
            self.workflow_status.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 7: Created {len(state.newsletter_sections)} newsletter sections")

            status_msg = f"✅ Step 7 completed successfully! Organized content into {len(state.newsletter_sections)} sections with {articles_assigned} articles assigned."
            status_msg += f"\n📑 Sections: {', '.join(state.newsletter_sections.keys())}"
            status_msg += f"\n💾 Section plan stored in persistent state. Current step: {state.current_step}"
            return status_msg

        except Exception as e:
            self.workflow_status.error_step(step_name, str(e))
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
    """Tool for Step 8: Draft section content"""

    def __init__(self, workflow_status: WorkflowStatus, verbose: bool = False):
        self.workflow_status = workflow_status
        self.verbose = verbose

    async def _draft_sections(self, ctx, args: str) -> str:
        """Execute Step 8: Draft Sections using persistent state"""
        step_name = "step_08_draft_sections"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.current_step >= 8:
            drafted_sections = [s for s in state.newsletter_sections.values() if s.get('content')]
            total_words = sum(len(s.get('content', '').split()) for s in drafted_sections)
            return f"Step 8 already completed! Drafted {len(drafted_sections)} sections with {total_words} total words."

        # Check if step 7 is completed
        if state.current_step < 7:
            return f"❌ Cannot execute Step 8: Step 7 (Select Sections) must be completed first. Current step: {state.current_step}"

        try:
            # Update workflow status for UI tracking
            self.workflow_status.start_step(step_name)

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

                    # Get the actual summary from state if available
                    summary_bullets = state.article_summaries.get(article_url, [
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

            # Update persistent state
            state.current_step = 8

            # Also update workflow status for UI
            self.workflow_status.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 8: Drafted {sections_drafted} sections")

            status_msg = f"✅ Step 8 completed successfully! Drafted {sections_drafted} sections with {total_words} total words."
            status_msg += f"\n📝 Average words per section: {avg_words_per_section:.0f}"
            status_msg += f"\n💾 Section content stored in persistent state. Current step: {state.current_step}"
            return status_msg

        except Exception as e:
            self.workflow_status.error_step(step_name, str(e))
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
    """Tool for Step 9: Finalize complete newsletter"""

    def __init__(self, workflow_status: WorkflowStatus, verbose: bool = False):
        self.workflow_status = workflow_status
        self.verbose = verbose

    async def _finalize_newsletter(self, ctx, args: str) -> str:
        """Execute Step 9: Finalize Newsletter using persistent state"""
        step_name = "step_09_finalize_newsletter"

        # Access the persistent state
        state: NewsletterAgentState = ctx.context

        # Check if step already completed via persistent state
        if state.current_step >= 9:
            newsletter_length = len(state.final_newsletter.split()) if state.final_newsletter else 0
            sections_count = len([s for s in state.newsletter_sections.values() if s.get('content')])
            return f"Step 9 already completed! Newsletter finalized with {sections_count} sections and {newsletter_length} words."

        # Check if step 8 is completed
        if state.current_step < 8:
            return f"❌ Cannot execute Step 9: Step 8 (Draft Sections) must be completed first. Current step: {state.current_step}"

        try:
            # Update workflow status for UI tracking
            self.workflow_status.start_step(step_name)

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

            # Mark workflow as complete
            state.current_step = 9
            state.workflow_complete = True

            # Also update workflow status for UI
            self.workflow_status.complete_step(step_name)

            if self.verbose:
                print(f"✅ Completed Step 9: Finalized newsletter ({newsletter_length} words)")

            status_msg = f"🎉 Step 9 completed successfully! Newsletter finalized with {sections_included} sections and {newsletter_length} words."
            status_msg += f"\n⭐ Quality score: {final_quality_score:.1f}/10"
            status_msg += f"\n📰 Complete newsletter stored in persistent state"
            status_msg += f"\n✅ Workflow complete! All 9 steps finished successfully."
            return status_msg

        except Exception as e:
            self.workflow_status.error_step(step_name, str(e))
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

    def __init__(self, session_id: str = "newsletter_agent", verbose: bool = False):
        """
        Initialize the NewsletterAgent with persistent state

        Args:
            session_id: Unique identifier for the session (for persistence)
            verbose: Enable verbose logging
        """
        # Initialize session for persistence
        self.session = SQLiteSession(session_id, "newsletter_agent.db")
        self.workflow_status = WorkflowStatus()  # Keep for progress tracking UI
        self.verbose = verbose

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

        # Create all workflow tools
        tools = self._create_workflow_tools()

        super().__init__(
            name="NewsletterAgent",
            instructions=system_prompt,
            model="gpt-4o-mini",
            tools=tools
        )

        # Initialize default state
        self.default_state = NewsletterAgentState()

        if self.verbose:
            print(f"Initialized NewsletterAgent with persistent state and 9-step workflow")
            print(f"Session ID: {session_id}")

    async def run_step(self, user_input: str) -> str:
        """Run a workflow step with persistent state"""
        result = await Runner.run(
            self,
            user_input,
            session=self.session,
            context=self.default_state,  # Will load from session if exists
            max_turns=50  # Increased for complete 9-step workflow
        )
        return result.final_output

    def _create_workflow_tools(self):
        """Create all workflow tools for the agent"""
        tools = []

        # Status checking and inspection tools
        workflow_status_tool = WorkflowStatusTool(self.workflow_status)
        tools.append(workflow_status_tool.create_tool())

        state_inspection_tool = StateInspectionTool(self.verbose)
        tools.append(state_inspection_tool.create_tool())

        # Workflow step tools - create FunctionTool instances
        gather_tool = GatherUrlsTool(self.workflow_status, self.verbose)
        tools.append(gather_tool.create_tool())

        filter_tool = FilterUrlsTool(self.workflow_status, self.verbose)
        tools.append(filter_tool.create_tool())

        download_tool = DownloadArticlesTool(self.workflow_status, self.verbose)
        tools.append(download_tool.create_tool())

        extract_tool = ExtractSummariesTool(self.workflow_status, self.verbose)
        tools.append(extract_tool.create_tool())

        cluster_tool = ClusterByTopicTool(self.workflow_status, self.verbose)
        tools.append(cluster_tool.create_tool())

        rate_tool = RateArticlesTool(self.workflow_status, self.verbose)
        tools.append(rate_tool.create_tool())

        sections_tool = SelectSectionsTool(self.workflow_status, self.verbose)
        tools.append(sections_tool.create_tool())

        draft_tool = DraftSectionsTool(self.workflow_status, self.verbose)
        tools.append(draft_tool.create_tool())

        finalize_tool = FinalizeNewsletterTool(self.workflow_status, self.verbose)
        tools.append(finalize_tool.create_tool())

        return tools


async def main():
    """Main function to create agent and run complete workflow"""
    print("🚀 Creating NewsletterAgent...")

    # Load environment variables like the notebook does
    dotenv.load_dotenv()

    # Set up OpenAI client with environment configuration
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    default_headers_str = os.getenv("OPENAI_DEFAULT_HEADERS")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Parse default headers if provided
    default_headers = {}
    if default_headers_str:
        try:
            default_headers = json.loads(default_headers_str)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse OPENAI_DEFAULT_HEADERS: {default_headers_str}")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url or "https://api.openai.com/v1",
        default_headers=default_headers
    )

    # Set default client for the agents SDK
    set_default_openai_client(client)

    print(f"OpenAI Base URL: {base_url or 'https://api.openai.com/v1'}")
    print(f"Default Headers: {default_headers}")

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