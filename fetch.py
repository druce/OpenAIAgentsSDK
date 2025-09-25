import os
import yaml
import json
import yaml
import time
import random
import logging

from pathlib import Path

from datetime import datetime, timedelta

import asyncio
import nest_asyncio
import aiohttp
import feedparser
import requests

import pydantic
from pydantic import BaseModel, Field, RootModel
from typing import Dict, TypedDict, Type, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

import openai
from openai import AsyncOpenAI

import agents
from agents.exceptions import InputGuardrailTripwireTriggered
from agents import (Agent, Runner, Tool, ModelSettings, FunctionTool, InputGuardrail, GuardrailFunctionOutput,
                    SQLiteSession, set_default_openai_api, set_default_openai_client
                   )

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from playwright.async_api import async_playwright

from prompt_loader import PromptLoader
from log_handler import SQLiteLogHandler, setup_sqlite_logging, sanitize_error_for_logging, log
# Removed utilities imports - functionality merged into newsletter_state.py
from config import DOWNLOAD_DIR
from scrape import get_browser, scrape_source, parse_source_file

class Fetcher:
    """
    Fetcher class for managing RSS, HTML, and API content retrieval.

    Provides a unified interface for fetching content from various sources
    with proper resource management and rate limiting.
    """

    def __init__(self, sources: Optional[Dict[str, Any]] = None, sources_file: str = "sources.yaml", max_concurrent: int = 8, logger: Optional[logging.Logger] = None):
        """
        Initialize the Fetcher with source configuration.

        Args:
            sources: Dictionary of source configurations (if None, loads from sources_file)
            sources_file: Path to sources.yaml file (used if sources is None)
            max_concurrent: Maximum number of concurrent requests
            logger: Optional logger instance (creates console logger if None)
        """
        self.logger = self._setup_logger(logger)
        # Load sources configuration
        if sources is None:
            # Load sources from YAML file
            try:
                self._log(f"Loading sources from {sources_file}", "fetcher_init", "INFO")
                with open(sources_file, 'r', encoding='utf-8') as file:
                    self.sources = yaml.safe_load(file) or {}

                # Log source breakdown
                rss_sources = [k for k, v in self.sources.items() if v.get('type') == 'rss']
                html_sources = [k for k, v in self.sources.items() if v.get('type') == 'html']
                api_sources = [k for k, v in self.sources.items() if v.get('type') == 'rest']

                self._log(f"Loaded {len(self.sources)} sources: {len(rss_sources)} RSS, {len(html_sources)} HTML, {len(api_sources)} API", "fetcher_init", "INFO")

                # Log individual sources for debugging
                for source_key, source_config in self.sources.items():
                    source_type = "RSS" if source_config.get('rss') else source_config.get('type', 'unknown')
                    self._log(f"Source '{source_key}': type={source_type}, url={source_config.get('url') or source_config.get('rss', 'N/A')}", "fetcher_sources", "DEBUG")
            except FileNotFoundError:
                self._log(f"Sources file not found: {sources_file}", "fetcher_init", "ERROR")
                raise FileNotFoundError(f'Sources file not found: {sources_file}')
            except yaml.YAMLError as e:
                self._log(f"Error parsing YAML: {str(e)}", "fetcher_init", "ERROR")
                raise ValueError(f'Error parsing YAML: {str(e)}')
        else:
            self.sources = sources
            self._log(f"Initialized with {len(self.sources)} provided sources", "fetcher_init", "INFO")

        self.max_concurrent = max_concurrent
        self._log(f"Fetcher initialized with max_concurrent={max_concurrent}", "fetcher_init", "INFO")
        self.session: Optional[aiohttp.ClientSession] = None
        # Browser context now managed by scrape.py module-level caching
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self) -> 'Fetcher':
        """Initialize resources when entering async context."""
        # Create aiohttp session
        self.session = aiohttp.ClientSession()

        # Browser context is now lazily initialized and cached at module level
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting async context."""
        # Close aiohttp session
        if self.session:
            await self.session.close()

        # Close playwright instance if it was created (browser context is cached at module level)
        if hasattr(self, '_playwright'):
            await self._playwright.stop()

    def _setup_logger(self, logger: Optional[logging.Logger] = None) -> logging.Logger:
        # Set up logger
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(f"fetcher_{id(self)}")
            self.logger.setLevel(logging.DEBUG)

            # Only add handler if none exist (to avoid duplicates)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.propagate = False
        return self.logger

    def _log(self, message: str, context: str = None, level: str = "INFO"):
        """Helper method for logging with context information."""
        level_map = {
            "DEBUG": self.logger.debug,
            "INFO": self.logger.info,
            "WARNING": self.logger.warning,
            "ERROR": self.logger.error,
            "CRITICAL": self.logger.critical
        }

        log_func = level_map.get(level.upper(), self.logger.info)

        if context:
            formatted_message = f"[{context}] {message}"
        else:
            formatted_message = message

        log_func(formatted_message)

    async def _get_browser_context(self):
        """Get browser context using scrape.py's enhanced get_browser() with caching."""
        # Initialize playwright if needed
        if not hasattr(self, '_playwright'):
            self._playwright = await async_playwright().start()

        # Use enhanced get_browser with caching - no need for our own caching logic
        return await get_browser(self._playwright, reuse=True)

    async def fetch_rss(self, source: str) -> Dict[str, Any]:
        """
        Fetch and parse RSS feed from a source record

        Args:
            source: Top-level key from sources.yaml (e.g., "Techmeme")

        Returns:
            Dict with source, results, status, metadata
        """
        # Check if session is initialized
        if not self.session:
            return {
                'source': source,
                'results': [],
                'status': 'error',
                'metadata': {'error': 'Fetcher must be used as async context manager: async with Fetcher() as f:'}
            }

        source_record = self.sources.get(source, {})
        rss_url = source_record.get('rss')
        if not rss_url:
            return {
                'source': source,
                'results': [],
                'status': 'error',
                'metadata': {'error': 'No RSS URL found in source record'}
            }

        try:
            self._log(f"Fetching RSS from {source}: {rss_url}", "fetch_rss", "INFO")
            timeout = aiohttp.ClientTimeout(total=10)
            async with self.session.get(rss_url, timeout=timeout) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)

                    # Extract articles from feed entries
                    articles = []
                    for entry in feed.entries[:50]:  # Limit to 50 entries
                        # Format title with title_detail if available
                        title = entry.get('title', '')
                        title_detail = entry.get('title_detail', {}).get('value', '') if entry.get('title_detail') else ''
                        formatted_title = f"{title}: {title_detail}" if title_detail and title_detail != title else title

                        article = {
                            'source': source,
                            'title': formatted_title,
                            'url': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'rss_summary': entry.get('summary', '') or entry.get('description', '')
                        }
                        articles.append(article)

                    self._log(f"RSS fetch successful for {source}: {len(articles)} articles", "fetch_rss", "INFO")
                    return {
                        'source': source,
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
                        'source': source,
                        'results': [],
                        'status': 'error',
                        'metadata': {'error': f'HTTP {response.status} from {rss_url}'}
                    }

        except Exception as e:
            self._log(f"RSS fetch failed for {source}: {str(e)}", "fetch_rss", "ERROR")
            return {
                'source': source,
                'results': [],
                'status': 'error',
                'metadata': {'error': f'Failed to fetch RSS: {str(e)}'}
            }

    async def fetch_html(self, source_key: str, do_download: bool = True) -> Dict[str, Any]:
        """
        Fetch and parse HTML source to extract article links

        Args:
            source_key: Top-level key from sources.yaml

        Returns:
            Dict with source_key, results, status, metadata
        """
        # Check if session is initialized (needed for browser context)
        if not self.session:
            return {
                'source': source_key,
                'results': [],
                'status': 'error',
                'metadata': {'error': 'Fetcher must be used as async context manager: async with Fetcher() as f:'}
            }

        source_dict = self.sources.get(source_key, {})
        url = source_dict.get('url')
        if not url:
            return {
                'source': source_key,
                'results': [],
                'status': 'error',
                'metadata': {'error': 'No URL found in source record'}
            }

        try:
            if do_download:
                self._log(f"Fetching HTML from {source_key}: {url}", "fetch_html", "INFO")
                # Get browser context
                browser_context = await self._get_browser_context()
                # Check if browser context is valid
                if not browser_context:
                    return {
                        'source': source_key,
                        'results': [],
                        'status': 'error',
                        'metadata': {'error': 'Browser context is not available or has been closed'}
                    }

                self._log(f"Source dict for {source_key}: {source_dict}", "fetch_html", "INFO")

                # Fetch the landing page HTML
                source_dict["sourcename"] = source_key
                _, file_path = await scrape_source(source_dict, browser_context, logger=self.logger)

            else:
                self._log(f"Using existing HTML file from {source_key}: {url}", "fetch_html", "INFO")
                filename = self.sources.get(source_key, {}).get('filename')
                source_dict["sourcename"] = source_key
                file_path = f'{DOWNLOAD_DIR}/{filename}.html'

            if not file_path:
                self._log(f"Failed to download HTML page from {source_key}: {url}", "fetch_html", "ERROR")
                return {
                    'source': source_key,
                    'results': [],
                    'status': 'error',
                    'metadata': {'error': 'Failed to download HTML page'}
                }

            # Parse the HTML file to extract article links
            source_dict['latest'] = file_path  # parse_file expects this key
            self._log(f"Parsing HTML file: {file_path}", "fetch_html", "INFO")
            link_list = parse_source_file(source_dict)
            self._log(f"Parsed HTML file: {file_path}", "fetch_html", "INFO")

            # Convert to same format as fetch_rss results
            articles = []
            for link in link_list:
                article = {
                    'source': source_key,
                    'title': link.get('title', ''),
                    'url': link.get('url', '')
                    # Note: No 'published' key for HTML sources
                }
                articles.append(article)

            self._log(f"HTML fetch successful for {source_key}: {len(articles)} articles", "fetch_html", "INFO")
            return {
                'source': source_key,
                'results': articles,
                'status': 'success',
                'metadata': {
                    'landing_page': url,
                    'articles_found': len(articles),
                    'file_path': file_path
                }
            }

        except Exception as e:
            self._log(f"HTML fetch failed for {source_key}: {str(e)}", "fetch_html", "ERROR")
            return {
                'source': source_key,
                'results': [],
                'status': 'error',
                'metadata': {'error': f'Failed to fetch HTML: {str(e)}'}
            }

    async def fetch_api(self, source_key: str) -> Dict[str, Any]:
        """
        Fetch from REST API sources by calling the appropriate function

        Args:
            source_key: Top-level key from sources.yaml

        Returns:
            Dict with source_key, results, status, metadata
        """
        source_record = self.sources.get(source_key, {})
        function_name = source_record.get('function_name')

        if not function_name:
            return {
                'source': source_key,
                'results': [],
                'status': 'error',
                'metadata': {'error': 'No function_name specified in source record'}
            }

        try:
            # Map function names to actual class methods
            function_map = {
                'fn_extract_newsapi': self.extract_newsapi,
                # Add more API functions here as needed
            }

            func = function_map.get(function_name)
            if not func:
                return {
                    'source': source_key,
                    'results': [],
                    'status': 'error',
                    'metadata': {'error': f'Unknown function: {function_name}'}
                }

            # Call the method and return its result
            # Note: Most API methods are synchronous, so we call them directly
            if asyncio.iscoroutinefunction(func):
                result = await func()
            else:
                result = func()

            # Ensure the source key matches our source_key
            result['source'] = source_key
            return result

        except Exception as e:
            return {
                'source': source_key,
                'results': [],
                'status': 'error',
                'metadata': {'error': f'Failed to fetch from API: {str(e)}'}
            }

    async def fetch_all(self, do_download: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch content from all sources concurrently
        uses semaphore from constructor to limit concurrent requests to max_concurrent

        Returns:
            List of results from all sources
        """
        self._log(f"Starting fetch_all for {len(self.sources)} sources", "fetch_all", "INFO")
        async def fetch_with_semaphore(source_key, source_record):
            async with self.semaphore:
                if source_record.get('type') == 'rss':
                    return await self.fetch_rss(source_key)
                elif source_record.get('type') == 'rest':
                    return await self.fetch_api(source_key)
                elif source_record.get('type') == 'html':
                    return await self.fetch_html(source_key, do_download)
                else:
                    return {
                        'source': source_key,
                        'results': [],
                        'status': 'error',
                        'metadata': {'error': 'Unknown source type: ' + source_record.get('type')}
                    }

        # Create tasks for all sources
        tasks = [
            fetch_with_semaphore(source_key, source_record)
            for source_key, source_record in self.sources.items()
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
                    'source': 'unknown',
                    'results': [],
                    'status': 'error',
                    'metadata': {'error': f'Exception during fetch: {str(result)}'}
                })

        # Log summary
        success_count = sum(1 for r in valid_results if r.get('status') == 'success')
        error_count = sum(1 for r in valid_results if r.get('status') == 'error')
        total_articles = sum(len(r.get('results', [])) for r in valid_results if r.get('status') == 'success')

        self._log(f"fetch_all completed: {success_count} successful, {error_count} failed, {total_articles} total articles", "fetch_all", "INFO")

        return valid_results

    def extract_newsapi(self) -> Dict[str, Any]:
        """
        Get AI news via newsapi - updated to return standard fetch format
        https://newsapi.org/docs/get-started
        """
        try:
            news_api_key = os.environ.get('NEWSAPI_API_KEY')
            if not news_api_key:
                return {
                    'source': 'NewsAPI',
                    'results': [],
                    'status': 'error',
                    'metadata': {'error': 'NEWSAPI_API_KEY environment variable not set'}
                }

            page_size = 100
            q = 'artificial intelligence'
            date_24h_ago = datetime.now() - timedelta(hours=24)
            formatted_date = date_24h_ago.strftime("%Y-%m-%dT%H:%M:%S")

            self._log(f"Fetching top {page_size} stories matching {q} since {formatted_date} from NewsAPI", "newsapi", "INFO")

            baseurl = 'https://newsapi.org/v2/everything'
            params = {
                'q': q,
                'from': formatted_date,
                'language': 'en',
                'sortBy': 'relevancy',
                'apiKey': news_api_key,
                'pageSize': page_size
            }

            response = requests.get(baseurl, params=params, timeout=60)
            if response.status_code != 200:
                return {
                    'source': 'NewsAPI',
                    'results': [],
                    'status': 'error',
                    'metadata': {'error': f'API call failed with status {response.status_code}: {response.text}'}
                }

            data = response.json()
            articles = []

            for article in data.get('articles', []):
                formatted_article = {
                    'source': 'NewsAPI',
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'published': article.get('publishedAt', '')
                }
                articles.append(formatted_article)

            return {
                'source': 'NewsAPI',
                'results': articles,
                'status': 'success',
                'metadata': {
                    'total_results': data.get('totalResults', 0),
                    'articles_returned': len(articles),
                    'query': q,
                    'date_from': formatted_date
                }
            }

        except Exception as e:
            self._log(f"Failed to fetch from NewsAPI: {str(e)}", "newsapi", "ERROR")
            return {
                'source': 'NewsAPI',
                'results': [],
                'status': 'error',
                'metadata': {'error': f'Failed to fetch from NewsAPI: {str(e)}'}
            }


async def gather_urls(sources_file: str = "sources.yaml", max_concurrent: int = 8) -> List[Dict[str, Any]]:
    """
    Load sources.yaml and fetch content from all sources concurrently using Fetcher class

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
            'source': 'error',
            'results': [],
            'status': 'error',
            'metadata': {'error': f'Sources file not found: {sources_file}'}
        }]
    except yaml.YAMLError as e:
        return [{
            'source': 'error',
            'results': [],
            'status': 'error',
            'metadata': {'error': f'Error parsing YAML: {str(e)}'}
        }]

    # Use Fetcher class for coordinated fetching
    async with Fetcher(sources, max_concurrent) as fetcher:
        return await fetcher.fetch_all()

