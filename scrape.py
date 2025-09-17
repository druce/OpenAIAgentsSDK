"""
Web scraping utilities.

This module contains lower level functions used for web scraping from web sites

https://github.com/AndyTheFactory/newspaper4k
https://github.com/fhamborg/news-please

"""
# flake8: noqa: E722
# pylint: disable=W0718  # bare-except
# pylint: disable=W1401  # backslash in RE

import asyncio
import re
import os
import logging
from urllib.parse import urljoin, urlparse
# import pdb
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
from dataclasses import dataclass
from time import monotonic

import random
import time
import datetime
from pathlib import Path
from dateutil import parser as date_parser

import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from playwright_stealth import Stealth
import tiktoken

import trafilatura

# from log_handler import log  # Replaced with standard logging
from config import (DOWNLOAD_DIR, IGNORE_LIST, PAGES_DIR, FIREFOX_PROFILE_PATH,  # SCREENSHOT_DIR,
                    MIN_TITLE_LEN, SLEEP_TIME, MAX_INPUT_TOKENS, DOMAIN_RATE_LIMIT,
                    SHORT_REQUEST_TIMEOUT)

# Module-level logger for default logging
# when calling from high level function, pass logger=logger but this is here as a default
_logger = logging.getLogger(__name__)

# Module-level browser context caching
_browser_context_cache = None
_browser_lock = None

@dataclass
class DomainState:
    """State tracking for individual domains in rate limiting.

    Maintains the timestamp of the last request and provides thread-safe
    access control for each domain to prevent hitting domains too frequently.
    Used by RateLimiter to enforce per-domain request throttling.
    """
    last_request: float = 0    # last request for a domain as monotonic timestamp
    lock: asyncio.Lock = None  # lock for concurrent access

    def __post_init__(self):
        if self.lock is None:
            self.lock = asyncio.Lock()

class RateLimiter:
    """Rate limiter for domain-based request throttling.

    Controls the frequency of requests to each domain by enforcing a delay
    between consecutive requests to the same domain. Each domain is tracked
    independently, allowing concurrent requests to different domains while
    respecting per-domain rate limits.

    Args:
        delay_seconds: Minimum delay in seconds between requests to the same domain.
                      Defaults to DOMAIN_RATE_LIMIT from config.
    """
    def __init__(self, delay_seconds: float = DOMAIN_RATE_LIMIT):
        self.delay = delay_seconds
        self.domains: Dict[str, DomainState] = defaultdict(DomainState)

    def can_proceed(self, domain: str) -> Tuple[bool, float]:
        """Check if domain can proceed, return (can_proceed, wait_time)"""
        state = self.domains[domain]
        now = monotonic()
        time_since_last = now - state.last_request

        if time_since_last >= self.delay:
            return True, 0.0
        else:
            return False, self.delay - time_since_last

    def mark_request(self, domain: str):
        """Mark that a request was made for this domain"""
        self.domains[domain].last_request = monotonic()

    async def try_acquire_domain_slot(self, domain: str) -> Tuple[bool, float]:
        """Atomically check and acquire domain slot if available (thread-safe)"""
        state = self.domains[domain]
        async with state.lock:
            now = monotonic()
            time_since_last = now - state.last_request

            if time_since_last >= self.delay:
                state.last_request = now  # Atomically mark request
                return True, 0.0
            else:
                return False, self.delay - time_since_last




def get_og_tags(source: str, logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    """
    scrapes Open Graph og: tags from a given URL or local file and returns them as a dictionary.

    Parameters:
    source (str): The URL of the webpage or the path to the local HTML file.
    logger (Optional[logging.Logger]): Optional logger for this operation.

    Returns:
    dict: A dictionary containing the og: tags found in the webpage.
    """
    logger = logger or _logger
    result_dict = {}
    content = None

    if source.startswith(("http://", "https://")):
        try:
            response = requests.get(source, timeout=SHORT_REQUEST_TIMEOUT)
            if response.status_code == 200:
                content = response.content
        except requests.RequestException as e:
            logger.error(f"Error scraping {source}: {e}")
    else:
        try:
            with open(source, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            logger.error(f"Error: File not found at {source}")
        except PermissionError as e:
            logger.error(f"Permission denied reading file {source}: {e}")
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading file {source}: {e}")
        except OSError as e:
            logger.error(f"OS error reading file {source}: {e}")
        except Exception as e:
            logger.error(f"Unknown error reading file {source}: {e}")

    if content:
        soup = BeautifulSoup(content, "html.parser")
        head = soup.head
        if head:
            og_tags = head.find_all(
                property=lambda prop: prop and prop.startswith("og:")
            )
            for tag in og_tags:
                if "content" in tag.attrs:
                    result_dict[tag["property"]] = tag["content"]

            page_title = ""
            title_tag = soup.find("title")
            if title_tag:
                page_title = title_tag.text
                if page_title:
                    result_dict["title"] = page_title

    return result_dict


def clean_url(link: Union[str, Any]) -> Optional[str]:
    """
    Trims everything in the link after a question mark such as a session ID.

    :param link: The input string or bs4 link.
    :return: The trimmed string.
    """
    if isinstance(link, str):
        s = link
    else:  # bs4 element
        try:
            s = link.get("href", "")
        except AttributeError:
            return ""

    # Find the position of the question mark
    question_mark_index = s.find("?")
    # If a question mark is found, trim the string up to that point
    if question_mark_index != -1:
        s = s[:question_mark_index]
    return s


def trunc_tokens(long_prompt: str, model: str = 'gpt-4o', maxtokens: int = MAX_INPUT_TOKENS) -> str:
    """return prompt string, truncated to maxtokens"""
    # Initialize the encoding for the model you are using, e.g., 'gpt-4'
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    # Encode the prompt into tokens, truncate, and return decoded prompt
    tokens = encoding.encode(long_prompt)
    if len(tokens) > maxtokens:
        tokens = tokens[:maxtokens]
        long_prompt = encoding.decode(tokens)
    return long_prompt


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a filename by removing unsafe characters and ensuring it is valid.
    E.g. take title and make it a valid filename

    Args:
        filename (str): The filename to sanitize.

    Returns:
        str: The sanitized filename.
    """
    sep = ""
    datestr = ""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove any other unsafe characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    # Remove leading or trailing underscores
    filename = filename.strip('_')
    # filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)
    trunc_len = 255 - len(datestr) - len(sep) - len(".html") - 1
    filename = filename[:trunc_len]
    return filename


def normalize_html(path: Path | str, logger: Optional[logging.Logger] = None) -> str:
    """
    Clean and extract text content from an HTML file, including titles and social media metadata.

    Args:
        path (Path | str): Path to the HTML file to process
        logger (Optional[logging.Logger]): Optional logger for this operation

    Returns:
        - str: Extracted and cleaned text content, or empty string if processing fails

    The function extracts:
        - Page title from <title> tag
        - Social media titles from OpenGraph and Twitter meta tags
        - Social media descriptions from OpenGraph and Twitter meta tags
        - Main content using trafilatura library

    All extracted content is concatenated and truncated to MAX_INPUT_TOKENS length.
    """
    logger = logger or _logger

    try:
        with open(path, 'r', encoding='utf-8') as file:
            html_content = file.read()
    except FileNotFoundError as exc:
        logger.error(f"File not found ({exc}): {path}")
        return ""
    except PermissionError as exc:
        logger.error(f"Permission denied reading {path}: {exc}")
        return ""
    except UnicodeDecodeError as exc:
        logger.error(f"Encoding error reading {path}: {exc}")
        return ""

    # Parse the HTML content using trafilatura
    soup = BeautifulSoup(html_content, 'html.parser')

    try:
        # Try to get the title from the <title> tag
        title_tag = soup.find("title")
        title_str = "Page title: " + title_tag.string.strip() + \
            "\n" if title_tag and title_tag.string else ""
    except (AttributeError, TypeError) as exc:
        title_str = ""
        logger.warning(f"Error extracting page title: {exc}")

    try:
        # Try to get the title from the Open Graph meta tag
        og_title_tag = soup.find("meta", property="og:title")
        if not og_title_tag:
            og_title_tag = soup.find(
                "meta", attrs={"name": "twitter:title"})
        og_title = og_title_tag["content"].strip(
        ) + "\n" if og_title_tag and og_title_tag.get("content") else ""
        og_title = "Social card title: " + og_title if og_title else ""
    except (AttributeError, KeyError, TypeError) as exc:
        og_title = ""
        logger.warning(f"Error extracting og:title: {exc}")

    try:
        # get summary from social media cards
        og_desc_tag = soup.find("meta", property="og:description")
        if not og_desc_tag:
            # Extract the Twitter description
            og_desc_tag = soup.find(
                "meta", attrs={"name": "twitter:description"})
        og_desc = og_desc_tag.get("content").strip() + \
            "\n" if og_desc_tag else ""
        og_desc = 'Social card description: ' + og_desc if og_desc else ""
    except (AttributeError, KeyError, TypeError) as exc:
        og_desc = ""
        logger.warning(f"Error extracting og:description: {exc}")

    # Get text and strip leading/trailing whitespace
    logger.debug(f"clean_html: {title_str + og_title + og_desc}")
    try:
        plaintext = trafilatura.extract(html_content)
        plaintext = plaintext.strip() if plaintext else ""
    except (ImportError, RuntimeError, ValueError) as exc:
        plaintext = html_content
        logger.warning(f"Trafilatura extraction failed: {exc}")

    # remove special tokens, have found in artiles about tokenization
    # All OpenAI special tokens follow the pattern <|something|>
    special_token_re = re.compile(r"<\|\w+\|>")
    plaintext = special_token_re.sub("", plaintext)
    visible_text = title_str + og_title + og_desc + plaintext
    visible_text = trunc_tokens(
        visible_text, model='gpt-4o', maxtokens=MAX_INPUT_TOKENS)
    return visible_text


async def get_browser(p: Any, reuse: bool = True) -> BrowserContext:
    """
    Initializes a Playwright browser context with stealth settings.

    Args:
        p (async_playwright.Playwright): The Playwright instance.
        reuse (bool): If True, reuse existing cached context; if False, create new one.

    Returns:
        BrowserContext: The initialized browser context.
    """
    global _browser_context_cache, _browser_lock

    if not reuse:
        # Create fresh browser - bypass cache
        return await _create_browser_context(p)

    # Fast path: if browser already exists and is valid, return it
    if _browser_context_cache is not None:
        try:
            # Quick test to see if context is still alive
            pages = _browser_context_cache.pages
            return _browser_context_cache
        except Exception:
            # Context is dead, clear cache
            _browser_context_cache = None

    # Initialize lock if needed
    if _browser_lock is None:
        _browser_lock = asyncio.Lock()

    # Use lock to prevent race conditions during initialization
    async with _browser_lock:
        # Double-check pattern: another coroutine might have initialized it while we waited
        if _browser_context_cache is None:
            _browser_context_cache = await _create_browser_context(p)

    return _browser_context_cache


async def _create_browser_context(p: Any) -> BrowserContext:
    """
    Internal function to create a new browser context with full configuration.

    Args:
        p (async_playwright.Playwright): The Playwright instance.

    Returns:
        BrowserContext: The initialized browser context.
    """
    viewport = random.choice([
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 1440, "height": 900},
        {"width": 1536, "height": 864},
        {"width": 1280, "height": 720}
    ])

    # random device-scale-factor for additional randomization
    device_scale_factor = random.choice([1, 1.25, 1.5, 1.75, 2])

    # Random color scheme and timezone
    color_scheme = random.choice(['light', 'dark', 'no-preference'])
    timezone_id = random.choice([
        'America/New_York', 'Europe/London', 'Europe/Paris',
        'Asia/Tokyo', 'Australia/Sydney', 'America/Los_Angeles'
    ])
    locale = random.choice([
        'en-US', 'en-GB'
    ])
    extra_http_headers = {
        "Accept-Language": f"{locale.split('-')[0]},{locale};q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "DNT": "1" if random.choice([True, False]) else "0"
    }

    b = await p.firefox.launch_persistent_context(
        user_data_dir=FIREFOX_PROFILE_PATH,
        headless=True,  # run without GUI
        viewport=viewport,
        device_scale_factor=device_scale_factor,
        timezone_id=timezone_id,
        color_scheme=color_scheme,
        extra_http_headers=extra_http_headers,
        # removes Playwright’s default flag
        ignore_default_args=["--enable-automation"],
        args=[
            # "--disable-blink-features=AutomationControlled",  # Chrome/Blink flag analogue
            "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"
        ],
        # Firefox preferences to ensure new windows open as tabs
        firefox_user_prefs={
            "browser.link.open_newwindow": 3,  # Open new windows in tabs instead
            "browser.link.open_newwindow.restriction": 0,  # Allow all redirections to tabs
            "browser.tabs.loadInBackground": False,  # Focus new tabs immediately
        },
        # provide a valid realistic User-Agent string for the latest Firefox on Apple Silicon
        # match OS / browser build
        user_agent="Mozilla/5.0 (Macintosh; ARM Mac OS X 14.4; rv:125.0) Gecko/20100101 Firefox/125.0",
        accept_downloads=True,
    )
    await apply_stealth_script(b)
    return b


async def apply_stealth_script(context: BrowserContext) -> None:
    """Apply stealth settings to a new page using playwright_stealth."""
    stealth = Stealth()
    page = await context.new_page()
    await stealth.apply_stealth_async(page)
    await page.close()


async def perform_human_like_actions(page: Page) -> Page:
    """Perform random human-like actions on the page to mimic real user behavior."""
    # Random mouse movements
    for _ in range(random.randint(3, 8)):
        # Move mouse with multiple steps to simulate human-like movement
        x = random.randint(100, 1200)
        y = random.randint(100, 700)
        steps = random.randint(5, 10)

        # Get current mouse position
        mouse_position = await page.evaluate("""() => {
            return {x: 0, y: 0}; // Default starting position
        }""")

        current_x = mouse_position.get('x', 0)
        current_y = mouse_position.get('y', 0)

        # Calculate increments for smooth movement
        for step in range(1, steps + 1):
            next_x = current_x + (x - current_x) * step / steps
            next_y = current_y + (y - current_y) * step / steps

            # Add slight randomness to path
            jitter_x = random.uniform(-5, 5)
            jitter_y = random.uniform(-5, 5)

            await page.mouse.move(next_x + jitter_x, next_y + jitter_y)
            await asyncio.sleep(random.uniform(0.01, 0.05))

    # Random scrolling behavior
    scroll_amount = random.randint(300, 700)
    await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
    await asyncio.sleep(random.uniform(0.5, 2))

    # Sometimes scroll back up a bit
    if random.random() > 0.7:
        await page.evaluate(f"window.scrollBy(0, -{random.randint(100, 300)})")
        await asyncio.sleep(random.uniform(0.3, 1))

    return page


async def scrape_urls_concurrent(
    urls: List[Tuple[int, str, str]],  # index, url, title
    concurrency: int = 16,
    rate_limit_seconds: float = DOMAIN_RATE_LIMIT,
    logger: Optional[logging.Logger] = None
) -> List[Tuple[int, str, str, str, str, Optional[str]]]:
    """
    Scrape a batch of URLs concurrently with worker pool and random URL selection.

    Uses a pool of concurrent workers that randomly select URLs from remaining set
    to prevent domain clustering. Each worker respects per-domain rate limiting.

    Args:
        urls: List of (index, url, title) tuples to scrape
        concurrency: Maximum number of concurrent worker tasks
        rate_limit_seconds: Seconds between requests to same domain
        logger: Optional logger for this operation

    Returns:
        List of (index, status, url, title, file_path, last_updated) results
        where status is 'success', 'ratelimit', or error description
    """
    logger = logger or _logger
    rate_limiter = RateLimiter(delay_seconds=rate_limit_seconds)
    semaphore = asyncio.Semaphore(concurrency)

    # Shared state for workers
    remaining_urls = set(urls)  # Set of (idx, url, title) tuples
    completed_results = {}  # Map idx -> result tuple
    total_urls = len(urls)
    processed_count = 0
    url_lock = asyncio.Lock()  # Protects remaining_urls set
    progress_lock = asyncio.Lock()  # Protects progress tracking

    async def scrape_single_url_concurrent(idx: int, url: str, title: str, browser: BrowserContext, rate_limiter: RateLimiter, logger: logging.Logger) -> Tuple[int, str, str, str, str, Optional[str]]:
        """Scrape a single URL concurrently with rate limiting check."""
        domain = urlparse(url).netloc

        # Check if file already exists
        title_sanitized = sanitize_filename(title)
        html_path = os.path.join(PAGES_DIR, f'{title_sanitized}.html')
        if os.path.exists(html_path):
            logger.info(f"File already exists: {html_path}")
            return (idx, 'success', url, title, html_path, None)

        # Skip ignored domains
        if domain in IGNORE_LIST:
            logger.info(f"Skipping ignored domain: {domain}")
            return (idx, 'success', url, title, "", None)

        # Atomically check and acquire domain slot - if blocked, return immediately
        can_proceed, wait_time = await rate_limiter.try_acquire_domain_slot(domain)
        if not can_proceed:
            logger.info(f"Rate limiting domain {domain}, will retry later (need to wait {wait_time:.1f}s)")
            return (idx, 'ratelimit', url, title, "", None)

        # Proceed with scraping (domain slot already acquired atomically)
        try:
            html_path, last_updated, final_url = await scrape_url(url, title, browser, logger=logger)
            return (idx, 'success', final_url or url, title, html_path or "", last_updated)

        except asyncio.TimeoutError as exc:
            error_msg = f"Timeout: {exc}"
            logger.error(f"Timeout scraping {url}: {exc}")
            return (idx, error_msg, url, title, "", None)
        except (ConnectionError, OSError) as exc:
            error_msg = f"Network error: {exc}"
            logger.error(f"Network error scraping {url}: {exc}")
            return (idx, error_msg, url, title, "", None)
        except Exception as exc:
            error_msg = f"Error: {exc}"
            logger.error(f"Unexpected error scraping {url}: {exc}")
            return (idx, error_msg, url, title, "", None)

    async def worker(worker_id: int, browser: BrowserContext) -> None:
        """Worker that continuously processes random URLs from the remaining set."""
        while True:
            # Get random URL from remaining set
            current_url_tuple = None
            async with url_lock:
                if not remaining_urls:
                    break  # No more URLs to process
                current_url_tuple = random.choice(list(remaining_urls))
                remaining_urls.remove(current_url_tuple)

            if current_url_tuple is None:
                break

            # Update progress counter and log
            async with progress_lock:
                nonlocal processed_count
                processed_count += 1
                current_progress = processed_count

            idx, url, title = current_url_tuple
            domain = urlparse(url).netloc

            logger.info(f"Worker {worker_id} fetching {current_progress} of {total_urls} {url}")

            # Process URL with semaphore limiting concurrency
            async with semaphore:
                result = await scrape_single_url_concurrent(idx, url, title, browser, rate_limiter, logger)

            if result[1] == 'ratelimit':  # result[1] is status
                # Put URL back in set for retry by another worker
                async with url_lock:
                    remaining_urls.add(current_url_tuple)

                # Decrement processed count since URL will be retried
                async with progress_lock:
                    processed_count -= 1

                logger.info(f"Worker {worker_id} re-queued rate-limited URL: {url}")

                # Add cooling-off delay to prevent tight loop when many same-domain URLs remain
                await asyncio.sleep(2.0)

            else:
                # Success or permanent error - store result
                async with progress_lock:
                    completed_results[idx] = result

                logger.info(f"Worker {worker_id} completed {url} with status: {result[1]}")

    # Concurrent processing with worker pool
    async with async_playwright() as p:
        logger.info(f"Launching browser for {len(urls)} URLs with {concurrency} concurrent workers")
        browser = await get_browser(p, reuse=True)

        # Create and start worker tasks
        workers = []
        for worker_id in range(concurrency):
            worker_task = asyncio.create_task(worker(worker_id, browser))
            workers.append(worker_task)

        # Wait for all workers to complete
        await asyncio.gather(*workers)

        logger.info("Closing browser")
        await browser.close()

    # Convert completed_results dict to sorted list by original index
    all_results = [completed_results[idx] for idx in sorted(completed_results.keys())]

    success_count = sum(1 for r in all_results if r[1] == 'success')
    error_count = len(all_results) - success_count
    logger.info(f"Completed scraping {len(all_results)} URLs: {success_count} successful, {error_count} failed")

    return all_results


# potentially switch to chromium, didn't do in the past due to chromedriver version issues but not an issue with playwright
# 1. test running a chrome.py with playwright and playwright-stealth and chromium, make a new profile, figure out what it uses
# 2. headless, add stealth options, log in to eg feedly using the profile, see that it works, where profile is
# 3. get your organic chrome user agent string and paste
# 4. migrate existing profile settings and test. now we have a good profile.
# 5. update get_browser below to use chrome and new profile. potentially ask o3 to look at your code and suggest a good stealth calling template.


async def scrape_url(url: str,
                    title: str,
                    browser_context: Optional[BrowserContext] = None,
                    click_xpath: Optional[str] = None,
                    scrolls: int = 0,
                    scroll_div: str = "",
                    initial_sleep: float = SLEEP_TIME,
                    destination: str = PAGES_DIR,
                    logger: Optional[logging.Logger] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    scrapes a URL using a Playwright browser context.

    Args:
        url (str): The URL to scrape.
        title (str): The title for the scraped page.
        click_xpath (str): An optional XPath expression to click on before saving.
        scrolls (int): The number of times to scroll to the bottom of the page and wait for new content to load.
        browser_context (BrowserContext): The Playwright browser context to use. If not provided, a new browser context will be initialized.
        initial_sleep (float): The number of seconds to wait after the page has loaded before clicking.
        destination (str): The directory to save the downloaded file.
        logger (logging.Logger): Optional logger for this operation.

    Returns:
        tuple: (html_path, last_updated_time, final_url) where:
            html_path (str): The path to the downloaded file.
            last_updated_time (str or None): The last update time of the page.
            final_url (str): The final URL after any redirects.

    # should add retry functionality, re-enable screenshots
    """
    logger = logger or _logger
    logger.info(f"scrape_url({url})")
    try:
        # make output directories
        logger.info(f"scraping {url} to {destination}")
        if not os.path.exists(destination):
            os.makedirs(destination)

        title = sanitize_filename(title)
        html_path = os.path.join(destination, f'{title}.html')
        # check if file already exists, don't re-download
        if os.path.exists(html_path):
            logger.info(f"File already exists: {html_path}")
            return html_path, None, url

        # if file does not exist, download
        logger.info(f"Downloading {url}")
        page = await browser_context.new_page()
        response = await page.goto(url, timeout=SHORT_REQUEST_TIMEOUT*1000, wait_until='domcontentloaded')
        logger.info(f"Response: {response.status}")
        logger.debug(f"Initial sleep: {initial_sleep}")
        sleep_time = initial_sleep + random.uniform(1, 3)
        await asyncio.sleep(sleep_time)
        await perform_human_like_actions(page)
        logger.debug("performed human like actions")
        if click_xpath:
            await asyncio.sleep(initial_sleep + random.uniform(1, 3))
            logger.info(f"Attempting to click on {click_xpath}")
            # click_xpath == '//*[@aria-label="Artificial intelligence"]'
            await page.wait_for_selector(f'xpath={click_xpath}')
            await page.click(f'xpath={click_xpath}')
        for i in range(scrolls):
            logger.info(f"Scrolling {title} ({i+1}/{scrolls})")
            await asyncio.sleep(random.uniform(1, 3))  # Stealth delay
            if scroll_div:
                await page.evaluate("""
                    const el = document.querySelector('%s');
                    if (el) {
                        el.scrollTop = el.scrollHeight;
                    } else {
                        window.scrollTo(0, document.body.scrollHeight);
                    }
                    """ % scroll_div)
            else:
                await page.evaluate('window.scrollTo(0, document.body.scrollHeight);')

        html_source = await page.content()
        if page.url != url:
            logger.info(f"Page URL redirected from {url} to {page.url}")
        # Determine last updated time, first try meta tags
        last_updated = None
        soup_meta = BeautifulSoup(html_source, "html.parser")
        meta_selectors = [
            ("property", "article:published_time"),
            ("property", "og:published_time"),
            ("property", "article:modified_time"),
            ("property", "og:updated_time"),
            ("name", "pubdate"),
            ("name", "publish_date"),
            ("name", "Last-Modified"),
            ("name", "lastmod"),
        ]
        for attr, val in meta_selectors:
            tag = soup_meta.find("meta", attrs={attr: val})
            if tag and tag.get("content"):
                last_updated = tag["content"]
                logger.debug(
                    f"Found last updated time from meta tag {attr}={val}: {last_updated}")
                break

        # if not last_updated:
        #     time_tag = soup_meta.find("time", datetime=True)
        #     if time_tag and time_tag.get("datetime"):
        #         last_updated = time_tag["datetime"]
        #         log(f"Found last updated time from time tag: {last_updated}")

        # for substack
        # Find all JSON-LD script blocks
        for script in soup_meta.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if data.get('@type') == 'NewsArticle':
                    last_updated = data.get('datePublished')
                    logger.debug(
                        f"Found script last updated time from script datePublished: {last_updated}")
                    break
            except Exception:
                continue

        # Check HTTP Last-Modified header
        if not last_updated:
            if response and response.headers.get("last-modified"):
                last_updated = response.headers.get("last-modified")
                logger.debug(
                    f"Found last updated time from HTTP header: {last_updated}")

        # Fallback to document.lastModified
        if not last_updated:
            try:
                last_updated = await page.evaluate("document.lastModified")
                logger.debug(
                    f"Found last updated time from document.lastModified: {last_updated}")
            except Exception:
                last_updated = None

        # Validate and normalize last_updated to Zulu datetime
        if last_updated and isinstance(last_updated, str):
            try:
                logger.debug(f"Attempting to parse last_updated: '{last_updated}' (type: {type(last_updated)})")
                dt = date_parser.parse(last_updated)
                logger.debug(f"Parsed datetime: {dt}")
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                    logger.debug(f"Added UTC timezone: {dt}")
                dt_utc = dt.astimezone(datetime.timezone.utc)
                logger.debug(f"Converted to UTC: {dt_utc}")
                last_updated = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                logger.debug(f"Formatted last_updated: {last_updated}")
            except Exception as e:
                logger.warning(f"Could not parse last_updated '{last_updated}': {type(e).__name__}: {e}")
                # set to 1 day ago
                try:
                    last_updated = (datetime.datetime.now(
                        datetime.timezone.utc) - datetime.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    logger.debug(f"Set fallback last_updated: {last_updated}")
                except Exception as fallback_e:
                    logger.error(f"Failed to create fallback datetime: {type(fallback_e).__name__}: {fallback_e}")
                    last_updated = None

        # Save HTML
        logger.info(f"Saving HTML to {html_path}")
        # if the file already exists, overwrite it
        with open(html_path, 'w', encoding='utf-8') as file:
            file.write(html_source)

        # Save screenshot for video
        # screenshot_path = f"{SCREENSHOT_DIR}/{title}.png"
        # await page.screenshot(path=screenshot_path)
        # Get the final URL after any redirects
        # Try to get canonical URL from the HTML source
        canonical_tag = soup_meta.find("link", rel="canonical")
        if canonical_tag and canonical_tag.get("href"):
            final_url = canonical_tag["href"]
        else:
            final_url = page.url

        await page.close()

        return html_path, last_updated, final_url
    except asyncio.TimeoutError as exc:
        logger.error(f"Timeout error scraping {url}: {exc}")
        return None, None, None
    except (ConnectionError, OSError) as exc:
        logger.error(f"Network error scraping {url}: {exc}")
        return None, None, None
    except Exception as exc:
        logger.error(f"Unexpected error scraping {url}: {exc}")
        return None, None, None
    finally:
        # Ensure page is always closed
        if 'page' in locals() and page:
            try:
                await page.close()
            except Exception as exc:
                logger.warning(f"Error closing page for {url}: {exc}")


def parse_source_file(source_dict: Dict[str, Any], logger: Optional[logging.Logger] = None) -> List[Dict[str, str]]:
    """
    Parse a saved HTML file and return a list of dictionaries with title, url, src for each link in the file.

    Args:
        source_dict (dict): A dictionary containing the source information.
        logger (Optional[logging.Logger]): Optional logger for this operation.

    Returns:
        list: A list of dictionaries, where each dictionary represents a link in the HTML file.
              Each dictionary contains the following keys: 'title', 'url', 'src'.

    Raises:
        None

    """

    logger = logger or _logger
    sourcename = source_dict['sourcename']
    filename = f'{DOWNLOAD_DIR}/{source_dict["filename"]}.html'
    url = source_dict.get("url")
    exclude = source_dict.get("exclude")
    include = source_dict.get("include")
    minlength = source_dict.get("minlength", MIN_TITLE_LEN)
    logger.debug(f"minlength from source_dict: {repr(source_dict.get('minlength'))}, MIN_TITLE_LEN: {repr(MIN_TITLE_LEN)}, final minlength: {repr(minlength)}")
    # Ensure minlength is never None for comparison operations
    if minlength is None:
        logger.warning(f"minlength was None! source_dict minlength: {repr(source_dict.get('minlength'))}, MIN_TITLE_LEN: {repr(MIN_TITLE_LEN)}")
        minlength = 28  # Default minimum title length

    link_list = []

    try:
        # get contents
        with open(filename, "r", encoding="utf-8") as file:
            html_content = file.read()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
        logger.error(f"Error reading file {filename}: {e}")
        return []

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all <a> tags
    if soup:
        links = soup.find_all("a")
    else:
        logger.warning(f"Skipping {url}, unable to parse")
        return []
    logger.debug(f"found {len(links)} raw links")

    # drop empty text
    links = [link for link in links if link.get_text(strip=True)]
    # drop some ArsTechnica links that are just the number of comments and dupe the primary link
    links = [link for link in links if not re.match(
        r"^(\d+)$", link.get_text(strip=True))]

    # convert relative links to absolute links using base URL if present
    base_tag = soup.find('base')
    base_url = base_tag.get('href') if base_tag else url
    for link in links:
        link["href"] = urljoin(base_url, link.get('href', ""))

    # drop empty url path, i.e. url = toplevel domain
    def get_path_from_url(url: str) -> str:
        return urlparse(url).path

    links = [link for link in links if len(
        get_path_from_url(clean_url(link))) > 1]
    # drop anything that is not http, like javascript: or mailto:
    links = [link for link in links if link.get(
        "href") and link.get("href").startswith("http")]

    if exclude:
        for pattern in exclude:
            # filter links by exclusion pattern
            links = [
                link
                for link in links
                if link.get("href") is not None and not re.match(pattern, link.get("href"))
            ]

    if include:
        for pattern in include:
            new_links = []
            for link in links:
                href = link.get("href")
                if href and re.match(pattern, href):
                    new_links.append(link)
            links = new_links

    for link in links:
        url = clean_url(link)
        title = link.get_text(strip=True)
        if title == "LINK":
            # try to update title if the title is LINK (or eventually other patterns)
            og_dict = get_og_tags(url, logger=logger)
            if og_dict.get("og:title"):
                title = og_dict.get("og:title")

        # skip some low quality links that don't have full headline, like link to a Twitter or Threads account
        if len(title) <= minlength and title != "LINK":
            continue

        link_list.append({"title": title, "url": url, "src": sourcename})

    logger.debug(f"found {len(link_list)} filtered links")

    return link_list


async def scrape_source(source_dict: Dict[str, Any], browser_context: Optional[BrowserContext] = None, logger: Optional[logging.Logger] = None) -> Tuple[str, Optional[str]]:
    """
    scrapes a landing page using scrape_url and parameters defined in sources.yaml.
    source_dict is the landing page parameters loaded from sources.yaml.
    Updates source_dict['latest'] with the path to the downloaded file.

    Args:
        source_dict (dict): A dictionary containing the parameters defined in sources.yaml.
        browser_context (BrowserContext, optional): The Playwright browser context to use. If not provided, a new browser context will be initialized.
        logger (logging.Logger, optional): Optional logger for this operation.

    Returns:
        str: The path to the downloaded file.

    Raises:
        Exception: If there is an error during the execution of the function.

    """
    logger = logger or _logger
    url = source_dict.get("url")
    filename = source_dict["filename"]
    sourcename = source_dict["sourcename"]
    click_xpath = source_dict.get("click", "")
    scrolls = source_dict.get("scroll", 0)
    scroll_div = source_dict.get("scroll_div", "")
    initial_sleep = source_dict.get("initial_sleep", SLEEP_TIME)

    logger.info(f"Starting scrape_source {url}, {filename}")

    # Open the page and scrape the HTML
    file_path, _, _ = await scrape_url(url, filename, browser_context,
                                      click_xpath, scrolls, scroll_div, initial_sleep, destination=DOWNLOAD_DIR, logger=logger)
    source_dict['latest'] = file_path
    return (sourcename, file_path)