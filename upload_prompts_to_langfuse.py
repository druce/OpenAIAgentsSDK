#!/usr/bin/env python3
"""
Script to upload all prompts from prompts.md to Langfuse.
This will create or update prompts in your Langfuse org/project.
"""

import os
import re
import json
import sys
import argparse
from pathlib import Path
import dotenv
import langfuse

# Load environment variables
dotenv.load_dotenv()


def parse_prompts_md(filepath: str = "prompts.md"):
    """
    Parse prompts.md file and extract all prompt information.

    Returns:
        list: List of prompt dictionaries with all metadata
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by prompt sections (marked by # Prompt: `name`)
    prompt_sections = re.split(r'\n---\n\n# Prompt: `([^`]+)`\n', content)[1:]  # Skip header

    prompts = []

    # Process pairs of (prompt_name, prompt_content)
    for i in range(0, len(prompt_sections), 2):
        if i + 1 >= len(prompt_sections):
            break

        prompt_name = prompt_sections[i]
        section_content = prompt_sections[i + 1]

        # Skip if it's the code references or summary section
        if prompt_name.startswith('Code References') or prompt_name.startswith('Summary'):
            continue

        # Extract metadata
        metadata_match = re.search(
            r'## Metadata\n- \*\*Version\*\*: (\S+)\n- \*\*Type\*\*: (\S+)\n- \*\*Labels\*\*: ([^\n]+)\n- \*\*Tags\*\*: ([^\n]+)',
            section_content
        )

        if not metadata_match:
            print(f"⚠️  Warning: Could not parse metadata for {prompt_name}")
            continue

        version = metadata_match.group(1)
        prompt_type = metadata_match.group(2) if metadata_match.group(2) != 'None' else None
        labels_str = metadata_match.group(3)
        tags_str = metadata_match.group(4)

        # Parse labels
        labels = []
        if labels_str and labels_str != 'None':
            labels = [l.strip() for l in labels_str.split(',')]

        # Parse tags
        tags = []
        if tags_str and tags_str != 'None':
            tags = [t.strip() for t in tags_str.split(',')]

        # Extract configuration
        config = {}
        config_match = re.search(
            r'## Configuration\n```json\n(.*?)\n```',
            section_content,
            re.DOTALL
        )
        if config_match:
            try:
                config = json.loads(config_match.group(1))
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Could not parse config for {prompt_name}: {e}")

        # Extract system prompt
        system_prompt = None
        system_match = re.search(
            r'## System Prompt\n```markdown\n(.*?)\n```',
            section_content,
            re.DOTALL
        )
        if system_match:
            system_prompt = system_match.group(1).strip()

        # Extract user prompt
        user_prompt = None
        user_match = re.search(
            r'## User Prompt\n```markdown\n(.*?)\n```',
            section_content,
            re.DOTALL
        )
        if user_match:
            user_prompt = user_match.group(1).strip()

        # Build prompt object
        prompt_data = {
            'name': prompt_name,
            'version': int(version) if version.isdigit() else 1,
            'type': prompt_type,
            'labels': labels,
            'tags': tags,
            'config': config,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt
        }

        prompts.append(prompt_data)

    return prompts


def upload_prompt_to_langfuse(client, prompt_data, dry_run=False):
    """
    Upload a single prompt to Langfuse.

    Args:
        client: Langfuse client instance
        prompt_data: Dictionary containing prompt information
        dry_run: If True, only simulate the upload

    Returns:
        bool: Success status
    """
    try:
        # Build the prompt messages array
        prompt_messages = []

        if prompt_data['system_prompt']:
            prompt_messages.append({
                'role': 'system',
                'content': prompt_data['system_prompt']
            })

        if prompt_data['user_prompt']:
            prompt_messages.append({
                'role': 'user',
                'content': prompt_data['user_prompt']
            })

        # If no prompts found, skip
        if not prompt_messages:
            print(f"⚠️  Skipping {prompt_data['name']}: No prompt content")
            return False

        if dry_run:
            print(f"✓ [DRY RUN] Would upload: {prompt_data['name']} (v{prompt_data['version']})")
            print(f"    Messages: {len(prompt_messages)}, Config: {prompt_data['config']}")
            return True

        # Create the prompt in Langfuse as a chat prompt
        # This ensures both system and user messages are stored as chat records
        client.create_prompt(
            name=prompt_data['name'],
            type="chat",  # CRITICAL: Must specify 'chat' type for chat prompts
            prompt=prompt_messages,
            config=prompt_data['config'],
            labels=prompt_data['labels'],
            tags=prompt_data['tags'] if prompt_data['tags'] else None
        )

        print(f"✓ Uploaded: {prompt_data['name']} (v{prompt_data['version']})")
        return True

    except Exception as e:
        print(f"✗ Failed to upload {prompt_data['name']}: {e}")
        return False


def main():
    """Main function to parse prompts.md and upload to Langfuse."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Upload prompts from prompts.md to Langfuse',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (no actual upload)
  python upload_prompts_to_langfuse.py --dry-run

  # Upload to Langfuse
  python upload_prompts_to_langfuse.py

  # Upload without confirmation
  python upload_prompts_to_langfuse.py --yes
        """
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate upload without actually uploading to Langfuse'
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Skip confirmation prompt'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Langfuse Prompt Upload Tool")
    if args.dry_run:
        print("[DRY RUN MODE - No uploads will be performed]")
    print("=" * 70)

    # Check environment variables
    has_public_key = bool(os.getenv('LANGFUSE_PUBLIC_KEY'))
    has_secret_key = bool(os.getenv('LANGFUSE_SECRET_KEY'))
    has_base_url = bool(os.getenv('LANGFUSE_BASE_URL'))

    print("\n📋 Environment Check:")
    print(f"  • LANGFUSE_PUBLIC_KEY: {'✓ Set' if has_public_key else '✗ Not set'}")
    print(f"  • LANGFUSE_SECRET_KEY: {'✓ Set' if has_secret_key else '✗ Not set'}")
    print(f"  • LANGFUSE_BASE_URL: {'✓ Set' if has_base_url else '✗ Not set'} {('(' + os.getenv('LANGFUSE_BASE_URL') + ')') if has_base_url else ''}")

    if not (has_public_key and has_secret_key):
        print("\n❌ Error: Missing required Langfuse credentials")
        print("   Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in your .env file")
        return

    # Initialize Langfuse client
    print("\n🔌 Connecting to Langfuse...")
    try:
        client = langfuse.Langfuse()
        print("✓ Connected successfully")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return

    # Parse prompts from prompts.md
    print("\n📖 Parsing prompts.md...")
    try:
        prompts = parse_prompts_md()
        print(f"✓ Found {len(prompts)} prompts")
    except FileNotFoundError:
        print("✗ prompts.md file not found")
        return
    except Exception as e:
        print(f"✗ Error parsing prompts.md: {e}")
        return

    if not prompts:
        print("\n❌ No prompts found to upload")
        return

    # Confirm upload
    if args.dry_run:
        print(f"\n🧪 DRY RUN: Would upload {len(prompts)} prompts to Langfuse")
    else:
        print(f"\n⚠️  This will upload {len(prompts)} prompts to Langfuse")

    print(f"   Target: {os.getenv('LANGFUSE_BASE_URL', 'https://cloud.langfuse.com')}")
    print("\nPrompts to upload:")
    for p in prompts:
        print(f"  • {p['name']} (v{p['version']})")

    if not args.yes and not args.dry_run:
        response = input("\n❓ Continue with upload? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("❌ Upload cancelled")
            return

    # Upload prompts
    if args.dry_run:
        print(f"\n🧪 DRY RUN: Simulating upload of {len(prompts)} prompts...")
    else:
        print(f"\n🚀 Uploading {len(prompts)} prompts...")
    print("-" * 70)

    success_count = 0
    fail_count = 0

    for prompt_data in prompts:
        if upload_prompt_to_langfuse(client, prompt_data, dry_run=args.dry_run):
            success_count += 1
        else:
            fail_count += 1

    # Flush the client to ensure all requests are sent (skip in dry run)
    if not args.dry_run:
        print("\n📤 Flushing Langfuse client...")
        client.flush()

    # Summary
    print("-" * 70)
    if args.dry_run:
        print(f"\n✅ Dry Run Complete!")
        print(f"  • Would upload: {success_count}")
        print(f"  • Would fail: {fail_count}")
        print(f"  • Total: {len(prompts)}")
        print(f"\n💡 Run without --dry-run to perform actual upload")
    else:
        print(f"\n✅ Upload Complete!")
        print(f"  • Successful: {success_count}")
        print(f"  • Failed: {fail_count}")
        print(f"  • Total: {len(prompts)}")

        if success_count > 0:
            print(f"\n🎉 Successfully uploaded {success_count} prompts to Langfuse!")
            print(f"   View them at: {os.getenv('LANGFUSE_BASE_URL', 'https://cloud.langfuse.com')}")


if __name__ == "__main__":
    main()
