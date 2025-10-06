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


def print_detailed_prompt_info(client, prompt_names: Set[str], all_prompts: Dict[str, dict]):
    """Print detailed information about each prompt."""

    print("\n" + "=" * 80)
    print("DETAILED PROMPT INFORMATION FROM LANGFUSE")
    print("=" * 80)

    sorted_prompts = sorted(prompt_names)

    for i, prompt_name in enumerate(sorted_prompts, 1):
        print(f"\n{'─' * 80}")
        print(f"[{i}] {prompt_name}")
        print('─' * 80)

        # Check if we have API metadata
        if prompt_name in all_prompts:
            metadata = all_prompts[prompt_name]
            print(f"  📊 Metadata (from API):")
            print(f"     • Version: {metadata.get('version', 'N/A')}")
            print(f"     • Type: {metadata.get('type', 'N/A')}")
            print(f"     • Labels: {metadata.get('labels', [])}")
            print(f"     • Tags: {metadata.get('tags', [])}")
            print(f"     • Created: {metadata.get('created_at', 'N/A')}")
            print(f"     • Updated: {metadata.get('updated_at', 'N/A')}")

        # Fetch full prompt details
        details = fetch_prompt_details(client, prompt_name)

        if 'error' in details:
            print(f"  ⚠️  Error fetching details: {details['error']}")
            continue

        print(f"\n  🔧 Configuration:")
        if details.get('config'):
            config_json = json.dumps(details['config'], indent=2)
            for line in config_json.split('\n'):
                print(f"     {line}")
        else:
            print(f"     • No configuration found")

        print(f"\n  💬 Prompt Content:")
        if details.get('prompt_content'):
            for idx, msg in enumerate(details['prompt_content']):
                print(f"\n     [{msg['role'].upper()}] ({msg['content_length']} characters)")
                print(f"     {'-' * 70}")
                # Print full content with indentation
                for line in msg['content'].split('\n'):
                    print(f"     {line}")
                print(f"     {'-' * 70}")
        else:
            print(f"     • No prompt content found")


def print_code_references(prompts_found: Dict[str, List[str]]):
    """Print which files use which prompts."""

    print("\n" + "=" * 80)
    print("CODE REFERENCES")
    print("=" * 80)

    for file_path, prompts in prompts_found.items():
        print(f"\n📁 {file_path}")
        for prompt in prompts:
            print(f"   • {prompt}")


def print_summary(prompt_names: Set[str], prompts_found: Dict[str, List[str]], all_prompts: Dict[str, dict]):
    """Print summary statistics."""

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n📊 Statistics:")
    print(f"   • Total unique prompts found in code: {len(prompt_names)}")
    print(f"   • Total files with prompt references: {len(prompts_found)}")
    print(f"   • Total prompts available in Langfuse API: {len(all_prompts)}")

    # Find prompts in Langfuse but not used in code
    if all_prompts:
        unused_prompts = set(all_prompts.keys()) - prompt_names
        if unused_prompts:
            print(f"\n⚠️  Prompts in Langfuse but not referenced in code ({len(unused_prompts)}):")
            for prompt in sorted(unused_prompts):
                print(f"   • {prompt}")


def main():
    """Main function to run the prompt discovery and detailed reporting."""

    print("🔍 Searching for Langfuse prompts in the project...")
    print()

    # Check environment variables
    has_public_key = bool(os.getenv('LANGFUSE_PUBLIC_KEY'))
    has_secret_key = bool(os.getenv('LANGFUSE_SECRET_KEY'))
    has_host = bool(os.getenv('LANGFUSE_HOST'))

    print(f"Environment variables:")
    print(f"  • LANGFUSE_PUBLIC_KEY: {'✓ Set' if has_public_key else '✗ Not set'}")
    print(f"  • LANGFUSE_SECRET_KEY: {'✓ Set' if has_secret_key else '✗ Not set'}")
    print(f"  • LANGFUSE_HOST: {'✓ Set' if has_host else '✗ Not set'} {('(' + os.getenv('LANGFUSE_HOST') + ')') if has_host else ''}")
    print()

    # Initialize Langfuse client
    try:
        client = langfuse.get_client()
        print("✓ Connected to Langfuse API\n")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize Langfuse client: {e}")
        print("   Continuing with code-based prompt discovery only...\n")
        client = None

    # Find prompts referenced in code
    prompts_found = find_langfuse_prompts_in_code()

    if not prompts_found:
        print("❌ No Langfuse prompts found in the project code.")
        if not client:
            return

    # Collect all unique prompt names from code
    all_prompt_names = set()
    for prompts in prompts_found.values():
        all_prompt_names.update(prompts)

    # Fetch all prompts from Langfuse API
    all_prompts_metadata = {}
    if client:
        print("📥 Fetching prompt list from Langfuse API...")
        all_prompts_metadata = get_all_prompts_from_langfuse(client)

        # Add any prompts from API that we haven't seen in code
        if all_prompts_metadata:
            all_prompt_names.update(all_prompts_metadata.keys())

    if not all_prompt_names:
        print("❌ No prompts found in code or Langfuse API.")
        return

    # Print code references
    if prompts_found:
        print_code_references(prompts_found)

    # Print detailed information for each prompt
    if client:
        print_detailed_prompt_info(client, all_prompt_names, all_prompts_metadata)

    # Print summary
    print_summary(all_prompt_names, prompts_found, all_prompts_metadata)

    print("\n" + "=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
