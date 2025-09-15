#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Tool for Web Scraping

This module implements an MCP server that provides web scraping capabilities
using the scrape.py functions. It offers tools for navigating URLs, extracting
content, capturing screenshots, and batch processing.

Tools provided:
- navigate_url: Navigate to a URL and save HTML
- extract_text: Extract plaintext content from a URL
- get_metadata: Extract Open Graph and metadata from a URL
- take_screenshot: Capture a screenshot of a webpage
- parse_links: Extract all links from a webpage
- batch_fetch: Process multiple URLs concurrently
"""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# MCP imports
try:
    from mcp import types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
except ImportError:
    print("MCP library not available. Install with: pip install mcp")
    exit(1)

# Local scraping imports
from scrape import (
    scrape_url, scrape_urls_concurrent, get_og_tags, normalize_html,
    parse_source_file, get_browser, sanitize_filename
)
from playwright.async_api import async_playwright
from config import DOWNLOAD_DIR, PAGES_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Server instance
app = Server("web-scraper")

# Tool definitions
NAVIGATE_URL_TOOL = types.Tool(
    name="navigate_url",
    description="Navigate to a URL and save the HTML content to a file",
    inputSchema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to navigate to"
            },
            "title": {
                "type": "string",
                "description": "Optional title for the saved file (defaults to URL)"
            },
            "wait_time": {
                "type": "number",
                "description": "Seconds to wait after page load (default: 5)",
                "minimum": 0,
                "maximum": 30
            },
            "scrolls": {
                "type": "integer",
                "description": "Number of page scrolls to perform (default: 0)",
                "minimum": 0,
                "maximum": 10
            }
        },
        "required": ["url"]
    }
)

EXTRACT_TEXT_TOOL = types.Tool(
    name="extract_text",
    description="Extract plaintext content from a URL or existing HTML file",
    inputSchema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to extract text from"
            },
            "html_file": {
                "type": "string",
                "description": "Path to existing HTML file (alternative to URL)"
            },
            "fetch_fresh": {
                "type": "boolean",
                "description": "Force fresh fetch even if file exists (default: false)"
            }
        },
        "oneOf": [
            {"required": ["url"]},
            {"required": ["html_file"]}
        ]
    }
)

GET_METADATA_TOOL = types.Tool(
    name="get_metadata",
    description="Extract Open Graph metadata and other structured data from a URL",
    inputSchema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to extract metadata from"
            },
            "include_text_preview": {
                "type": "boolean",
                "description": "Include a text preview in the response (default: false)"
            }
        },
        "required": ["url"]
    }
)

TAKE_SCREENSHOT_TOOL = types.Tool(
    name="take_screenshot",
    description="Capture a screenshot of a webpage",
    inputSchema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to screenshot"
            },
            "width": {
                "type": "integer",
                "description": "Viewport width (default: 1920)",
                "minimum": 320,
                "maximum": 3840
            },
            "height": {
                "type": "integer",
                "description": "Viewport height (default: 1080)",
                "minimum": 240,
                "maximum": 2160
            },
            "full_page": {
                "type": "boolean",
                "description": "Capture full page height (default: false)"
            },
            "wait_time": {
                "type": "number",
                "description": "Seconds to wait before screenshot (default: 3)",
                "minimum": 0,
                "maximum": 30
            }
        },
        "required": ["url"]
    }
)

PARSE_LINKS_TOOL = types.Tool(
    name="parse_links",
    description="Extract all links from a webpage with filtering options",
    inputSchema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to parse links from"
            },
            "include_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Regex patterns for links to include"
            },
            "exclude_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Regex patterns for links to exclude"
            },
            "min_title_length": {
                "type": "integer",
                "description": "Minimum link text length (default: 3)",
                "minimum": 1
            }
        },
        "required": ["url"]
    }
)

BATCH_FETCH_TOOL = types.Tool(
    name="batch_fetch",
    description="Fetch multiple URLs concurrently with rate limiting",
    inputSchema={
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "title": {"type": "string"}
                    },
                    "required": ["url", "title"]
                },
                "description": "List of URLs and titles to fetch"
            },
            "concurrency": {
                "type": "integer",
                "description": "Max concurrent downloads (default: 4)",
                "minimum": 1,
                "maximum": 10
            },
            "rate_limit": {
                "type": "number",
                "description": "Seconds between requests to same domain (default: 2.0)",
                "minimum": 0.1,
                "maximum": 10.0
            }
        },
        "required": ["urls"]
    }
)

# Tool handlers

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls."""
    try:
        if name == "navigate_url":
            return await handle_navigate_url(arguments)
        elif name == "extract_text":
            return await handle_extract_text(arguments)
        elif name == "get_metadata":
            return await handle_get_metadata(arguments)
        elif name == "take_screenshot":
            return await handle_take_screenshot(arguments)
        elif name == "parse_links":
            return await handle_parse_links(arguments)
        elif name == "batch_fetch":
            return await handle_batch_fetch(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [types.TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]

async def handle_navigate_url(args: Dict[str, Any]) -> List[types.TextContent]:
    """Navigate to URL and save HTML."""
    url = args["url"]
    title = args.get("title", urlparse(url).netloc)
    wait_time = args.get("wait_time", 5)
    scrolls = args.get("scrolls", 0)

    logger.info(f"Navigating to {url}")

    async with async_playwright() as p:
        browser = await get_browser(p)
        try:
            html_path, last_updated, final_url = await scrape_url(
                url=url,
                title=title,
                browser_context=browser,
                scrolls=scrolls,
                initial_sleep=wait_time,
                destination=PAGES_DIR,
                logger=logger
            )

            if html_path:
                file_size = os.path.getsize(html_path) if os.path.exists(html_path) else 0
                result = {
                    "status": "success",
                    "url": url,
                    "final_url": final_url,
                    "html_path": html_path,
                    "file_size_bytes": file_size,
                    "last_updated": last_updated
                }
                return [types.TextContent(
                    type="text",
                    text=f"Successfully navigated to {url}\n" +
                         f"Final URL: {final_url}\n" +
                         f"Saved to: {html_path}\n" +
                         f"File size: {file_size:,} bytes\n" +
                         f"Last updated: {last_updated or 'Unknown'}\n\n" +
                         f"Full result: {json.dumps(result, indent=2)}"
                )]
            else:
                return [types.TextContent(
                    type="text",
                    text=f"Failed to navigate to {url}"
                )]
        finally:
            await browser.close()

async def handle_extract_text(args: Dict[str, Any]) -> List[types.TextContent]:
    """Extract plaintext from URL or HTML file."""
    url = args.get("url")
    html_file = args.get("html_file")
    fetch_fresh = args.get("fetch_fresh", False)

    if url:
        # First navigate to URL if needed
        title = urlparse(url).netloc
        title_sanitized = sanitize_filename(title)
        html_path = os.path.join(PAGES_DIR, f'{title_sanitized}.html')

        if not os.path.exists(html_path) or fetch_fresh:
            async with async_playwright() as p:
                browser = await get_browser(p)
                try:
                    html_path, _, _ = await scrape_url(
                        url=url,
                        title=title,
                        browser_context=browser,
                        destination=PAGES_DIR,
                        logger=logger
                    )
                finally:
                    await browser.close()

        if not html_path or not os.path.exists(html_path):
            return [types.TextContent(
                type="text",
                text=f"Failed to fetch or find HTML for {url}"
            )]
    else:
        html_path = html_file

    # Extract text content
    if not os.path.exists(html_path):
        return [types.TextContent(
            type="text",
            text=f"HTML file not found: {html_path}"
        )]

    text_content = normalize_html(html_path, logger=logger)
    word_count = len(text_content.split())
    char_count = len(text_content)

    return [types.TextContent(
        type="text",
        text=f"Extracted text from: {html_path}\n" +
             f"Characters: {char_count:,}\n" +
             f"Words: {word_count:,}\n\n" +
             f"Content:\n{text_content}"
    )]

async def handle_get_metadata(args: Dict[str, Any]) -> List[types.TextContent]:
    """Extract metadata from URL."""
    url = args["url"]
    include_preview = args.get("include_text_preview", False)

    logger.info(f"Extracting metadata from {url}")

    # Get Open Graph and other metadata
    metadata = get_og_tags(url, logger=logger)

    result = {
        "url": url,
        "metadata": metadata,
        "metadata_count": len(metadata)
    }

    response_text = f"Metadata extracted from: {url}\n" + \
                   f"Found {len(metadata)} metadata fields\n\n"

    if metadata:
        response_text += "Metadata:\n"
        for key, value in metadata.items():
            response_text += f"  {key}: {value}\n"
    else:
        response_text += "No metadata found\n"

    if include_preview:
        # Also extract a text preview
        title = urlparse(url).netloc
        title_sanitized = sanitize_filename(title)
        html_path = os.path.join(PAGES_DIR, f'{title_sanitized}.html')

        if not os.path.exists(html_path):
            async with async_playwright() as p:
                browser = await get_browser(p)
                try:
                    html_path, _, _ = await scrape_url(
                        url=url,
                        title=title,
                        browser_context=browser,
                        destination=PAGES_DIR,
                        logger=logger
                    )
                finally:
                    await browser.close()

        if html_path and os.path.exists(html_path):
            text_preview = normalize_html(html_path, logger=logger)[:500]
            response_text += f"\nText preview (first 500 chars):\n{text_preview}..."

    response_text += f"\n\nFull result: {json.dumps(result, indent=2)}"

    return [types.TextContent(type="text", text=response_text)]

async def handle_take_screenshot(args: Dict[str, Any]) -> List[types.TextContent]:
    """Capture screenshot of webpage."""
    url = args["url"]
    width = args.get("width", 1920)
    height = args.get("height", 1080)
    full_page = args.get("full_page", False)
    wait_time = args.get("wait_time", 3)

    logger.info(f"Taking screenshot of {url}")

    # Create screenshot filename
    domain = urlparse(url).netloc
    screenshot_name = f"screenshot_{sanitize_filename(domain)}.png"
    screenshot_path = os.path.join(DOWNLOAD_DIR, screenshot_name)

    async with async_playwright() as p:
        browser = await get_browser(p)
        try:
            page = await browser.new_page()
            await page.set_viewport_size({"width": width, "height": height})

            response = await page.goto(url, timeout=60000, wait_until='domcontentloaded')
            await asyncio.sleep(wait_time)

            # Take screenshot
            await page.screenshot(
                path=screenshot_path,
                full_page=full_page
            )

            await page.close()

            if os.path.exists(screenshot_path):
                file_size = os.path.getsize(screenshot_path)
                result = {
                    "status": "success",
                    "url": url,
                    "screenshot_path": screenshot_path,
                    "file_size_bytes": file_size,
                    "viewport": {"width": width, "height": height},
                    "full_page": full_page,
                    "response_status": response.status if response else None
                }
                return [types.TextContent(
                    type="text",
                    text=f"Screenshot captured successfully\n" +
                         f"URL: {url}\n" +
                         f"Saved to: {screenshot_path}\n" +
                         f"File size: {file_size:,} bytes\n" +
                         f"Viewport: {width}x{height}\n" +
                         f"Full page: {full_page}\n\n" +
                         f"Full result: {json.dumps(result, indent=2)}"
                )]
            else:
                return [types.TextContent(
                    type="text",
                    text=f"Failed to capture screenshot of {url}"
                )]

        finally:
            await browser.close()

async def handle_parse_links(args: Dict[str, Any]) -> List[types.TextContent]:
    """Parse links from webpage."""
    url = args["url"]
    include_patterns = args.get("include_patterns", [])
    exclude_patterns = args.get("exclude_patterns", [])
    min_title_length = args.get("min_title_length", 3)

    logger.info(f"Parsing links from {url}")

    # First fetch the page
    title = urlparse(url).netloc
    async with async_playwright() as p:
        browser = await get_browser(p)
        try:
            html_path, _, _ = await scrape_url(
                url=url,
                title=title,
                browser_context=browser,
                destination=PAGES_DIR,
                logger=logger
            )
        finally:
            await browser.close()

    if not html_path or not os.path.exists(html_path):
        return [types.TextContent(
            type="text",
            text=f"Failed to fetch page: {url}"
        )]

    # Parse links using scrape.py function
    source_dict = {
        "sourcename": title,
        "title": title,
        "latest": html_path,
        "url": url,
        "include": include_patterns if include_patterns else None,
        "exclude": exclude_patterns if exclude_patterns else None,
        "minlength": min_title_length
    }

    links = parse_source_file(source_dict, logger=logger)

    result = {
        "url": url,
        "total_links": len(links),
        "links": links
    }

    response_text = f"Parsed links from: {url}\n" + \
                   f"Found {len(links)} links\n\n"

    if links:
        response_text += "Links:\n"
        for i, link in enumerate(links[:20], 1):  # Show first 20 links
            response_text += f"{i:2d}. {link['title'][:60]}...\n     {link['url']}\n"

        if len(links) > 20:
            response_text += f"\n... and {len(links) - 20} more links\n"
    else:
        response_text += "No links found matching criteria\n"

    response_text += f"\n\nFull result: {json.dumps(result, indent=2)}"

    return [types.TextContent(type="text", text=response_text)]

async def handle_batch_fetch(args: Dict[str, Any]) -> List[types.TextContent]:
    """Batch fetch multiple URLs."""
    urls_data = args["urls"]
    concurrency = args.get("concurrency", 4)
    rate_limit = args.get("rate_limit", 2.0)

    logger.info(f"Batch fetching {len(urls_data)} URLs with concurrency={concurrency}")

    # Convert to format expected by fetch_urls_concurrent
    urls_list = [
        (i, item["url"], item["title"])
        for i, item in enumerate(urls_data)
    ]

    # Fetch all URLs concurrently
    results = await scrape_urls_concurrent(
        urls=urls_list,
        concurrency=concurrency,
        rate_limit_seconds=rate_limit,
        logger=logger
    )

    # Process results
    successful = 0
    failed = 0
    total_size = 0

    result_data = []
    for idx, url, title, file_path, last_updated in results:
        if file_path and os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            total_size += file_size
            successful += 1
            status = "success"
        else:
            file_size = 0
            failed += 1
            status = "failed"

        result_data.append({
            "index": idx,
            "url": url,
            "title": title,
            "status": status,
            "file_path": file_path,
            "file_size_bytes": file_size,
            "last_updated": last_updated
        })

    summary = {
        "total_urls": len(urls_data),
        "successful": successful,
        "failed": failed,
        "total_size_bytes": total_size,
        "concurrency": concurrency,
        "rate_limit_seconds": rate_limit,
        "results": result_data
    }

    response_text = f"Batch fetch completed\n" + \
                   f"Total URLs: {len(urls_data)}\n" + \
                   f"Successful: {successful}\n" + \
                   f"Failed: {failed}\n" + \
                   f"Total data: {total_size:,} bytes\n" + \
                   f"Concurrency: {concurrency}\n" + \
                   f"Rate limit: {rate_limit}s\n\n"

    response_text += "Results:\n"
    for result in result_data:
        status_icon = "✓" if result["status"] == "success" else "✗"
        size_str = f"{result['file_size_bytes']:,}b" if result['file_size_bytes'] > 0 else "0b"
        response_text += f"{status_icon} {result['title'][:50]:<50} {size_str:>10}\n"

    response_text += f"\n\nFull result: {json.dumps(summary, indent=2)}"

    return [types.TextContent(type="text", text=response_text)]

# List available tools
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        NAVIGATE_URL_TOOL,
        EXTRACT_TEXT_TOOL,
        GET_METADATA_TOOL,
        TAKE_SCREENSHOT_TOOL,
        PARSE_LINKS_TOOL,
        BATCH_FETCH_TOOL
    ]

async def main():
    """Run the MCP server."""
    # Ensure output directories exist
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(PAGES_DIR, exist_ok=True)

    logger.info("Starting Web Scraper MCP Server")
    logger.info(f"Download directory: {DOWNLOAD_DIR}")
    logger.info(f"Pages directory: {PAGES_DIR}")

    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())