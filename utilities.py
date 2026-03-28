from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import html
import os
from pathlib import Path
import re
import smtplib
from typing import List, Optional

import dotenv
import pandas as pd


# Shared font stack and colors matching druce.ai
_FONT = "'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif"
_NAVY = "#111155"
_AMBER = "#d2691e"
_CARD_BG = "#f6f6fc"
_TEXT = "#111111"
_GRAY = "#828282"
_BORDER_GRAY = "#b0b0b0"
_OUTER_BG = "#ededf0"
_INNER_BG = "#fdfdfd"
_RULE = "#e0e0e0"


def _parse_markdown_links(links_md: str) -> list:
    """Parse markdown links like '[Site](url) [Site2](url2)' into list of (name, url) tuples."""
    return re.findall(r'\[([^\]]+)\]\(([^)]+)\)', links_md)


def _make_pill(site_name: str) -> str:
    """Generate an inline source pill span."""
    escaped = html.escape(site_name)
    return (
        f'<span style="display:inline-block; padding:1px 6px; '
        f'border:1px solid {_BORDER_GRAY}; border-radius:4px; '
        f'font-size:11px; color:{_GRAY}; vertical-align:middle;">'
        f'{escaped}</span>'
    )


def _make_story_row(headline: str, links_md: str, url: Optional[str] = None) -> str:
    """Generate one story card row (table row with left border, headline, inline pills)."""
    escaped_headline = html.escape(headline)
    links = _parse_markdown_links(links_md)

    # Use first link URL if no explicit url provided
    if not url and links:
        url = links[0][1]

    # Build pills
    pills = " ".join(_make_pill(name) for name, _ in links)

    # Build headline (linked or plain)
    if url:
        headline_html = (
            f'<a href="{html.escape(url)}" style="font-weight:600; '
            f'color:{_TEXT}; text-decoration:none;">{escaped_headline}</a>'
        )
    else:
        headline_html = f'<span style="font-weight:600; color:{_TEXT};">{escaped_headline}</span>'

    return f"""<tr><td style="padding:0 0 10px 0;">
  <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="border-left:4px solid {_AMBER}; background-color:{_CARD_BG};">
  <tr><td style="padding:10px 14px 10px 20px; font-size:14px; line-height:1.5; font-family:{_FONT};">
    {headline_html}<br>{pills}
  </td></tr>
  </table>
</td></tr>"""


def _make_section_divider() -> str:
    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%">'
        f'<tr><td style="border-top:1px solid {_RULE}; padding-bottom:24px;"></td></tr></table>'
    )


def _make_narrative_row(narrative: str) -> str:
    """Generate an HTML row for a section narrative paragraph."""
    escaped = html.escape(narrative)
    return f"""<tr><td style="padding:0 0 14px 0; font-size:14px; line-height:1.6; color:#333333; font-family:{_FONT};">
    {escaped}
</td></tr>"""


def format_newsletter_email(
    title: str,
    date_str: str,
    newsletter_section_df: pd.DataFrame,
    site_name: str = "Skynet&Chill",
    site_url: str = "https://skynetandchill.com",
    section_narratives: Optional[dict] = None,
) -> str:
    """Format a newsletter DataFrame into a styled HTML email.

    Args:
        title: Newsletter title/headline
        date_str: Display date (e.g. "February 16, 2026")
        newsletter_section_df: DataFrame sorted by sort_order/rating with columns:
            cat, section_title, headline, links (markdown format), rating
        site_name: Brand name for the header bar
        site_url: URL for the header link and footer
        section_narratives: Optional dict mapping category name to a narrative
            paragraph string. When provided, the narrative is rendered above
            the bullet-point headlines in each section.

    Returns:
        Complete HTML email string ready to send
    """
    escaped_title = html.escape(title)
    escaped_site = html.escape(site_name)
    escaped_date = html.escape(date_str)

    # Deduplicate by story id before counting and rendering
    if 'id' in newsletter_section_df.columns:
        newsletter_section_df = newsletter_section_df.drop_duplicates(
            subset=['id'], keep='first')

    story_count = len(newsletter_section_df)
    source_count = set()
    for links_md in newsletter_section_df['links']:
        for name, _ in _parse_markdown_links(links_md):
            source_count.add(name)
    sources_str = f"{story_count} stories from {len(source_count)}+ sources"

    if section_narratives is None:
        section_narratives = {}

    # Build section HTML, skipping duplicate story IDs
    sections_html = ""
    last_cat = ""
    section_index = 0
    seen_ids = set()

    for row in newsletter_section_df.itertuples():
        # Skip duplicate story IDs
        story_id = getattr(row, 'id', None)
        if story_id is not None and story_id in seen_ids:
            continue
        if story_id is not None:
            seen_ids.add(story_id)

        if row.cat != last_cat:
            # Close previous section (add divider between sections)
            if last_cat:
                sections_html += "</table>\n" + _make_section_divider() + "\n"

            last_cat = row.cat
            section_index += 1
            section_title = html.escape(row.section_title)

            # Section header
            sections_html += (
                f'<table role="presentation" cellpadding="0" cellspacing="0" '
                f'border="0" width="100%" style="margin-bottom:32px;">\n'
                f'<tr><td style="font-size:19px; font-weight:600; color:{_NAVY}; '
                f'padding-bottom:14px; font-family:{_FONT};">'
                f'{section_title}</td></tr>\n'
            )

            # Insert narrative paragraph if available for this section
            if row.cat in section_narratives:
                sections_html += _make_narrative_row(section_narratives[row.cat]) + "\n"

        sections_html += _make_story_row(row.headline, row.links) + "\n"

    # Close final section
    if last_cat:
        sections_html += "</table>\n"

    # Assemble full email
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escaped_title}</title>
<!--[if mso]><style>table,td{{font-family:Arial,sans-serif!important;}}</style><![endif]-->
</head>
<body style="margin:0; padding:0; background-color:{_OUTER_BG}; font-family:{_FONT};">

<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="background-color:{_OUTER_BG};">
<tr><td align="center" style="padding:20px 12px;">

<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="680" style="max-width:680px; width:100%; background-color:{_INNER_BG};">

<!-- Header bar -->
<tr>
<td style="background-color:{_NAVY}; padding:16px 32px;">
  <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%">
  <tr>
    <td style="font-size:22px; font-weight:600; color:#ffffff; font-family:{_FONT};">
      <a href="{html.escape(site_url)}" style="color:#ffffff; text-decoration:none;">{escaped_site}</a>
    </td>
    <td align="right" style="font-size:13px; color:rgba(255,255,255,0.6); font-family:{_FONT};">
      AI News Digest
    </td>
  </tr>
  </table>
</td>
</tr>

<!-- Title -->
<tr>
<td style="padding:40px 32px 8px 32px;">
  <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%">
  <tr><td style="font-size:28px; font-weight:600; color:{_TEXT}; line-height:1.25; font-family:{_FONT};">{escaped_title}</td></tr>
  <tr><td style="padding-top:10px; font-size:13px; color:{_GRAY}; font-family:{_FONT};">{escaped_date} &nbsp;&middot;&nbsp; {sources_str}</td></tr>
  </table>
</td>
</tr>

<!-- Rule -->
<tr><td style="padding:20px 32px 0 32px;"><table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%"><tr><td style="border-top:1px solid {_RULE};"></td></tr></table></td></tr>

<!-- Content -->
<tr>
<td style="padding:24px 32px 16px 32px;">
{sections_html}
</td>
</tr>

<!-- Footer -->
<tr>
<td style="padding:0 32px 28px 32px;">
  <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="border-top:1px solid {_RULE};">
  <tr><td style="padding:20px 0 0 0; text-align:center;">
    <a href="{html.escape(site_url)}" style="font-size:14px; font-weight:600; color:{_NAVY}; text-decoration:none; font-family:{_FONT};">{html.escape(site_url.replace('https://', ''))}</a>
  </td></tr>
  <tr><td style="padding:8px 0 0 0; font-size:12px; color:{_GRAY}; text-align:center; font-family:{_FONT};">Generated on {escaped_date} by AI Newsletter Agent</td></tr>
  </table>
</td>
</tr>

</table>
</td></tr>
</table>

</body>
</html>"""


def send_email(
    to_addresses: List[str],
    subject: str,
    html_content: str,
    from_address: Optional[str] = None,
) -> None:
    """Send an HTML email via Gmail SMTP.

    Args:
        to_addresses: List of recipient email addresses
        subject: Email subject line
        html_content: Complete HTML content (full document or fragment)
        from_address: Sender address (defaults to GMAIL_USER env var)
    """
    email_address = from_address or os.getenv("GMAIL_USER")

    message = MIMEMultipart()
    message['From'] = email_address
    message['To'] = ", ".join(to_addresses)
    message['Subject'] = subject
    message.attach(MIMEText(html_content, 'html'))

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(email_address, os.getenv("GMAIL_PASSWORD"))
        server.sendmail(email_address, to_addresses, message.as_string())


def send_gmail(subject: str, html_str: str) -> None:
    """Send mail using gmail smtp server (legacy wrapper).

    Wraps send_email for backward compatibility. Sends to GMAIL_USER.
    """
    email_address = os.getenv("GMAIL_USER")
    send_email([email_address], subject, html_str)


def export_newsletter_html(html_content: str, output_dir: str = "out") -> Path:
    """Write newsletter HTML to disk with date-stamped filename and latest symlink."""
    from config import OUTPUT_DIR
    if output_dir == "out":
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = Path(output_dir) / f"{date_str}.html"
    filepath.write_text(html_content)
    latest = Path(output_dir) / "latest.html"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(filepath.name)
    return filepath


def validate_sources(sources_dict: dict) -> List[str]:
    """Validate sources configuration and return list of error/warning strings.

    Returns an empty list if everything is valid.
    Prefixes errors with 'ERROR:' and warnings with 'WARNING:'.
    """
    issues = []
    valid_types = {"rss", "html", "rest"}

    for name, config in sources_dict.items():
        if not isinstance(config, dict):
            issues.append(f"ERROR: Source '{name}' is not a dictionary")
            continue

        source_type = config.get("type")
        if not source_type:
            issues.append(f"ERROR: Source '{name}' missing required 'type' field")
            continue
        if source_type not in valid_types:
            issues.append(f"ERROR: Source '{name}' has invalid type '{source_type}' (must be one of {valid_types})")
            continue

        # Type-specific required fields
        if source_type == "rss" and not (config.get("rss") or config.get("url")):
            issues.append(f"ERROR: RSS source '{name}' needs 'rss' or 'url' field")
        if source_type == "html" and not config.get("url"):
            issues.append(f"ERROR: HTML source '{name}' needs 'url' field")
        if source_type == "rest" and not config.get("function_name"):
            issues.append(f"ERROR: REST source '{name}' needs 'function_name' field")

        # Validate regex patterns
        for field in ("include", "exclude"):
            patterns = config.get(field, [])
            if isinstance(patterns, list):
                for pattern in patterns:
                    try:
                        re.compile(pattern)
                    except re.error as e:
                        issues.append(f"ERROR: Source '{name}' has invalid regex in '{field}': {pattern!r} ({e})")

        # Validate numeric fields
        for field in ("scroll", "initial_sleep", "minlength"):
            value = config.get(field)
            if value is not None:
                if not isinstance(value, (int, float)) or value <= 0:
                    issues.append(f"WARNING: Source '{name}' field '{field}' should be a positive number, got {value!r}")

    return issues


def generate_run_summary(state) -> str:
    """Generate a run summary with per-step timings and article counts.

    Args:
        state: NewsletterAgentState with workflow steps

    Returns:
        HTML table string summarizing the run
    """
    rows = []
    for step in state.steps:
        if step.started_at and step.completed_at:
            total_secs = int((step.completed_at - step.started_at).total_seconds())
            mins, secs = divmod(total_secs, 60)
            duration_str = f"{mins}:{secs:02d}"
        elif step.started_at:
            duration_str = "running..."
        else:
            duration_str = "-"

        rows.append(
            f"<tr><td>{html.escape(step.name)}</td>"
            f"<td>{step.status.value}</td>"
            f"<td>{duration_str}</td></tr>"
        )

    articles_gathered = len(state.headline_data)
    summary_parts = [f"Articles gathered: {articles_gathered}"]

    # Count articles with summaries, ratings if available
    df_data = state.headline_data
    if df_data:
        with_summary = sum(1 for d in df_data if d.get("summary"))
        with_rating = sum(1 for d in df_data if d.get("rating") is not None)
        if with_summary:
            summary_parts.append(f"Summarized: {with_summary}")
        if with_rating:
            summary_parts.append(f"Rated: {with_rating}")

    sections_count = len(state.newsletter_section_data)
    if sections_count:
        summary_parts.append(f"Newsletter stories: {sections_count}")

    stats_html = " &middot; ".join(summary_parts)

    return (
        f"<p>{stats_html}</p>"
        f'<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;">'
        f"<tr><th>Step</th><th>Status</th><th>Duration</th></tr>"
        f"{''.join(rows)}"
        f"</table>"
    )


def print_run_summary(state) -> None:
    """Print a plain-text run summary to stdout."""
    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)

    for step in state.steps:
        if step.started_at and step.completed_at:
            total_secs = int((step.completed_at - step.started_at).total_seconds())
            mins, secs = divmod(total_secs, 60)
            duration_str = f"{mins}:{secs:02d}"
        elif step.started_at:
            duration_str = "running..."
        else:
            duration_str = "-"

        print(f"  {step.name:<25} {step.status.value:<12} {duration_str}")

    articles = len(state.headline_data)
    print(f"\n  Articles gathered: {articles}")
    if state.headline_data:
        with_summary = sum(1 for d in state.headline_data if d.get("summary"))
        with_rating = sum(1 for d in state.headline_data if d.get("rating") is not None)
        if with_summary:
            print(f"  Summarized: {with_summary}")
        if with_rating:
            print(f"  Rated: {with_rating}")
    if state.newsletter_section_data:
        print(f"  Newsletter stories: {len(state.newsletter_section_data)}")
    print("=" * 60)


if __name__ == "__main__":
    dotenv.load_dotenv()
    test_html = """
    <h1>Test Email</h1>
    <p>This is a <strong>test email</strong> sent from the utilities module.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
    </ul>
    <p>If you're reading this, the email was sent successfully!</p>
    """

    print("Sending test email...")
    send_gmail("Test Email from utilities.py", test_html)
    print("Test email sent successfully!")
