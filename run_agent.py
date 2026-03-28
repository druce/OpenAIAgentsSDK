#!/usr/bin/env python3
"""CLI entry point for the Newsletter Agent workflow."""

import argparse
import asyncio
import logging
import os
import time
import traceback
from datetime import datetime

import dotenv
import nest_asyncio

from config import DEFAULT_CONCURRENCY, NEWSAGENTDB, LOGDB
from log_handler import setup_sqlite_logging
from newsletter_state import NewsletterAgentState
from news_agent import NewsletterAgent
from utilities import send_gmail, validate_sources, generate_run_summary, print_run_summary

nest_asyncio.apply()


def _step_status_tag(state) -> str:
    """Return a short status tag derived from workflow steps.

    Returns 'All Steps Complete' when every step finished, otherwise
    '<Step Name> incomplete' for the first non-complete step.
    """
    if state and hasattr(state, 'steps') and state.steps:
        for step in state.steps:
            if step.status.value != "complete":
                return f"{step.name} incomplete"
        return "All Steps Complete"
    return "unknown status"


def send_run_notification(session_id, duration, success, state=None, error_msg=None, summary_html=""):
    """Send a brief email notification about the run outcome."""
    try:
        today = datetime.now().strftime("%B %d, %Y")
        tag = _step_status_tag(state)
        mins, secs = divmod(int(duration), 60)
        duration_str = f"{mins}:{secs:02d}"
        if success:
            subject = f"Newsletter {tag} - {today}"
            body = f"<p>Session <code>{session_id}</code> completed in {duration_str}.</p>{summary_html}"
        else:
            subject = f"Newsletter {tag} - {today}"
            body = f"<p>Session <code>{session_id}</code> failed after {duration_str}.</p><pre>{error_msg}</pre>{summary_html}"
        send_gmail(subject, body)
    except Exception:
        pass  # notification is best-effort


def parse_args():
    parser = argparse.ArgumentParser(
        description="Newsletter Agent - run the complete workflow from the command line"
    )
    parser.add_argument(
        "-n", "--nofetch", action="store_true", default=False,
        help="Skip web fetching, use existing downloaded sources",
    )
    parser.add_argument(
        "-r", "--reprocess-since", type=str, default=None,
        help="Force reprocessing articles after this datetime (YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Number of concurrent browser instances (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "-e", "--max-edits", type=int, default=2,
        help="Maximum summary rewrites (default: 2)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume a previous session ID instead of starting fresh",
    )
    parser.add_argument(
        "-s", "--step", type=str, default=None,
        help='Run a specific step (e.g. "filter_urls") instead of all steps',
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-t", "--timeout", type=int, default=30,
        help="OpenAI API timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--notify", action=argparse.BooleanOptionalAction, default=None,
        help="Send email notification on completion/failure (default: on unless --step)",
    )
    return parser.parse_args()


async def run(args):
    dotenv.load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY environment variable not set")

    # Session ID
    if args.resume:
        session_id = args.resume
        print(f"Resuming session: {session_id}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        session_id = f"newsletter_{timestamp}"
        print(f"New session: {session_id}")

    # Resolve notify default: on for full runs, off for single steps
    should_notify = args.notify if args.notify is not None else (args.step is None)

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_sqlite_logging("run_agent", db_path=LOGDB, level=log_level)

    # State
    do_download = not args.nofetch
    state = None
    if args.resume:
        # Try to load persisted state from the database
        probe = NewsletterAgentState(session_id=session_id, db_path=NEWSAGENTDB)
        state = probe.load_latest_from_db()
        if state:
            # Override runtime flags from CLI args
            state.do_download = do_download
            state.concurrency = args.concurrency
            state.max_edits = args.max_edits
            if args.reprocess_since:
                state.reprocess_since = args.reprocess_since
            current = state.get_current_step()
            completed = state.get_completed_steps()
            print(f"Loaded saved state: {len(completed)} steps completed, "
                  f"resuming from '{current or 'all complete'}'")
        else:
            print("WARNING: No saved state found for session, starting fresh")

    if state is None:
        state = NewsletterAgentState(
            session_id=session_id,
            db_path=NEWSAGENTDB,
            do_download=do_download,
            reprocess_since=args.reprocess_since,
            verbose=args.verbose,
            concurrency=args.concurrency,
            max_edits=args.max_edits,
        )

    # Validate sources configuration
    import yaml
    try:
        with open("sources.yaml", "r", encoding="utf-8") as f:
            sources_config = yaml.safe_load(f) or {}
        issues = validate_sources(sources_config)
        errors = [i for i in issues if i.startswith("ERROR:")]
        warnings = [i for i in issues if i.startswith("WARNING:")]
        for w in warnings:
            print(f"  {w}")
        if errors:
            for e in errors:
                print(f"  {e}")
            raise SystemExit(f"Source validation failed with {len(errors)} error(s)")
        if not issues:
            print(f"Validated {len(sources_config)} sources OK")
    except FileNotFoundError:
        print("WARNING: sources.yaml not found, skipping validation")

    # Agent
    agent = NewsletterAgent(
        session_id=session_id,
        state=state,
        verbose=args.verbose,
        logger=logger,
        timeout=float(args.timeout),
    )
    state.serialize_to_db("initialize")

    print(f"do_download={do_download}  concurrency={args.concurrency}  "
          f"max_edits={args.max_edits}  timeout={args.timeout}s")

    # Run
    start_time = time.time()
    duration = 0.0
    success = False
    error_msg = None
    summary_html = ""

    try:
        if args.step:
            print(f"Running single step: {args.step}")
            result = await agent.run_tool_direct(args.step)
        else:
            print("Running all remaining steps...")
            result = await agent.run_step("Run all the workflow steps in order and create the newsletter")

        duration = time.time() - start_time
        success = True

        print("=" * 80)
        print(f"Completed in {duration:.1f}s")
        print(result)
        print_run_summary(state)
        summary_html = generate_run_summary(state)
    except Exception as e:
        duration = time.time() - start_time
        error_msg = traceback.format_exc()
        summary_html = generate_run_summary(state)
        print("=" * 80)
        print(f"FAILED after {duration:.1f}s")
        print(error_msg)
        print_run_summary(state)
        raise
    finally:
        if should_notify:
            send_run_notification(session_id, duration, success, state=state, error_msg=error_msg, summary_html=summary_html)


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
