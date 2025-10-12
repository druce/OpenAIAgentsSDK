#!/usr/bin/env python3
"""
Script to list all Langfuse prompts used in the project.
Uses the Langfuse Python SDK to fetch detailed information about each prompt.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Set, List
import dotenv
import langfuse

# Load environment variables
dotenv.load_dotenv()


def find_langfuse_prompts_in_code(directory=".") -> Dict[str, List[str]]:
    """
    Find all Langfuse prompts referenced in Python files.

    Returns:
        dict: Dictionary mapping files to lists of prompt names found
    """
    prompt_pattern = r'get_prompt\(["\']([^"\']+)["\']\)'
    prompts_found = {}

    # Search through all Python files
    for py_file in Path(directory).rglob("*.py"):
        # Skip certain directories
        if any(skip in str(py_file) for skip in ['.git', '__pycache__', 'venv', '.venv']):
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find all prompt names in this file
            matches = re.findall(prompt_pattern, content)
            if matches:
                prompts_found[str(py_file)] = matches

        except Exception as e:
            print(f"Error reading {py_file}: {e}")

    return prompts_found


def get_all_prompts_from_langfuse(client) -> Dict[str, dict]:
    """
    Fetch all available prompts from Langfuse API.

    Returns:
        dict: Dictionary mapping prompt names to their metadata
    """
    # Note: The Langfuse client doesn't have a method to list all prompts
    # We'll rely on code-based discovery instead
    return {}


def fetch_prompt_details(client, prompt_name: str) -> dict:
    """
    Fetch detailed information about a specific prompt from Langfuse.

    Returns:
        dict: Detailed prompt information including content, config, and metadata
    """
    try:
        lf_prompt = client.get_prompt(prompt_name)

        details = {
            'name': prompt_name,
            'version': getattr(lf_prompt, 'version', None),
            'type': getattr(lf_prompt, 'type', None),
            'labels': getattr(lf_prompt, 'labels', []),
            'tags': getattr(lf_prompt, 'tags', []),
            'config': getattr(lf_prompt, 'config', {}),
            'prompt_content': [],
        }

        # Extract prompt content (system, user messages)
        if hasattr(lf_prompt, 'prompt') and isinstance(lf_prompt.prompt, list):
            for idx, msg in enumerate(lf_prompt.prompt):
                if isinstance(msg, dict):
                    role = msg.get('role', f'message_{idx}')
                    content = msg.get('content', '')

                    details['prompt_content'].append({
                        'role': role,
                        'content_length': len(content),
                        'content': content
                    })

        return details

    except Exception as e:
        return {
            'name': prompt_name,
            'error': str(e)
        }


def generate_markdown_output(client, prompt_names: Set[str], all_prompts: Dict[str, dict]) -> str:
    """Generate markdown documentation for all prompts."""

    markdown_lines = ["# Langfuse Prompts Documentation\n"]

    sorted_prompts = sorted(prompt_names)

    for i, prompt_name in enumerate(sorted_prompts, 1):
        markdown_lines.append(f"\n---\n")
        markdown_lines.append(f"\n# Prompt: `{prompt_name}`\n")

        # Fetch full prompt details
        details = fetch_prompt_details(client, prompt_name)

        if 'error' in details:
            markdown_lines.append(f"\n**⚠️ Error fetching details:** `{details['error']}`\n")
            continue

        # Metadata section
        markdown_lines.append(f"\n## Metadata\n")
        markdown_lines.append(f"- **Version**: {details.get('version', 'N/A')}\n")
        markdown_lines.append(f"- **Type**: {details.get('type', 'N/A')}\n")

        if details.get('labels'):
            markdown_lines.append(f"- **Labels**: {', '.join(details['labels'])}\n")
        else:
            markdown_lines.append(f"- **Labels**: None\n")

        if details.get('tags'):
            markdown_lines.append(f"- **Tags**: {', '.join(details['tags'])}\n")
        else:
            markdown_lines.append(f"- **Tags**: None\n")

        # Configuration section
        if details.get('config'):
            markdown_lines.append(f"\n## Configuration\n")
            markdown_lines.append("```json\n")
            markdown_lines.append(json.dumps(details['config'], indent=2))
            markdown_lines.append("\n```\n")

        # Prompt content sections
        if details.get('prompt_content'):
            for msg in details['prompt_content']:
                role = msg['role']
                content = msg['content']

                if role == 'system':
                    markdown_lines.append(f"\n## System Prompt\n")
                elif role == 'user':
                    markdown_lines.append(f"\n## User Prompt\n")
                else:
                    markdown_lines.append(f"\n## {role.capitalize()} Prompt\n")

                markdown_lines.append("```markdown\n")
                markdown_lines.append(content)
                markdown_lines.append("\n```\n")
        else:
            markdown_lines.append(f"\n**No prompt content found**\n")

    return ''.join(markdown_lines)


def generate_code_references_markdown(prompts_found: Dict[str, List[str]]) -> str:
    """Generate markdown for code references."""

    markdown_lines = ["\n---\n\n# Code References\n"]

    if not prompts_found:
        markdown_lines.append("\nNo code references found.\n")
        return ''.join(markdown_lines)

    markdown_lines.append("\nThe following prompts are referenced in these files:\n\n")

    for file_path, prompts in sorted(prompts_found.items()):
        markdown_lines.append(f"## `{file_path}`\n\n")
        for prompt in prompts:
            markdown_lines.append(f"- `{prompt}`\n")
        markdown_lines.append("\n")

    return ''.join(markdown_lines)


def generate_summary_markdown(prompt_names: Set[str], prompts_found: Dict[str, List[str]], all_prompts: Dict[str, dict]) -> str:
    """Generate summary statistics in markdown."""

    markdown_lines = ["\n---\n\n# Summary\n\n"]

    markdown_lines.append(f"- **Total unique prompts found in code**: {len(prompt_names)}\n")
    markdown_lines.append(f"- **Total files with prompt references**: {len(prompts_found)}\n")
    markdown_lines.append(f"- **Total prompts available in Langfuse API**: {len(all_prompts)}\n")

    # Find prompts in Langfuse but not used in code
    if all_prompts:
        unused_prompts = set(all_prompts.keys()) - prompt_names
        if unused_prompts:
            markdown_lines.append(f"\n## ⚠️ Unused Prompts\n\n")
            markdown_lines.append(f"Prompts in Langfuse but not referenced in code ({len(unused_prompts)}):\n\n")
            for prompt in sorted(unused_prompts):
                markdown_lines.append(f"- `{prompt}`\n")

    return ''.join(markdown_lines)


def main():
    """Main function to run the prompt discovery and generate markdown documentation."""

    import sys

    # Write diagnostic info to stderr so it doesn't pollute markdown output
    sys.stderr.write("🔍 Searching for Langfuse prompts in the project...\n")

    # Check environment variables
    has_public_key = bool(os.getenv('LANGFUSE_PUBLIC_KEY'))
    has_secret_key = bool(os.getenv('LANGFUSE_SECRET_KEY'))
    has_host = bool(os.getenv('LANGFUSE_HOST'))

    sys.stderr.write(f"\nEnvironment variables:\n")
    sys.stderr.write(f"  • LANGFUSE_PUBLIC_KEY: {'✓ Set' if has_public_key else '✗ Not set'}\n")
    sys.stderr.write(f"  • LANGFUSE_SECRET_KEY: {'✓ Set' if has_secret_key else '✗ Not set'}\n")
    sys.stderr.write(f"  • LANGFUSE_HOST: {'✓ Set' if has_host else '✗ Not set'} {('(' + os.getenv('LANGFUSE_HOST') + ')') if has_host else ''}\n\n")

    # Initialize Langfuse client
    try:
        client = langfuse.get_client()
        sys.stderr.write("✓ Connected to Langfuse API\n\n")
    except Exception as e:
        sys.stderr.write(f"⚠️  Warning: Could not initialize Langfuse client: {e}\n")
        sys.stderr.write("   Continuing with code-based prompt discovery only...\n\n")
        client = None

    # Find prompts referenced in code
    prompts_found = find_langfuse_prompts_in_code()

    if not prompts_found:
        sys.stderr.write("❌ No Langfuse prompts found in the project code.\n")
        if not client:
            return

    # Collect all unique prompt names from code
    all_prompt_names = set()
    for prompts in prompts_found.values():
        all_prompt_names.update(prompts)

    # Fetch all prompts from Langfuse API
    all_prompts_metadata = {}
    if client:
        sys.stderr.write("📥 Fetching prompt details from Langfuse API...\n")
        all_prompts_metadata = get_all_prompts_from_langfuse(client)

        # Add any prompts from API that we haven't seen in code
        if all_prompts_metadata:
            all_prompt_names.update(all_prompts_metadata.keys())

    if not all_prompt_names:
        sys.stderr.write("❌ No prompts found in code or Langfuse API.\n")
        return

    # Generate markdown output
    markdown_output = []

    # Generate detailed prompt documentation
    if client:
        sys.stderr.write(f"📝 Generating markdown for {len(all_prompt_names)} prompts...\n")
        markdown_output.append(generate_markdown_output(client, all_prompt_names, all_prompts_metadata))

    # Add code references
    if prompts_found:
        markdown_output.append(generate_code_references_markdown(prompts_found))

    # Add summary
    markdown_output.append(generate_summary_markdown(all_prompt_names, prompts_found, all_prompts_metadata))

    # Write markdown to stdout
    print(''.join(markdown_output))

    sys.stderr.write("\n✓ Analysis complete! Markdown written to stdout.\n")
    sys.stderr.write("   Usage: python list_langfuse_prompts.py > prompts.md\n")


if __name__ == "__main__":
    main()
