# Guide: Upload Prompts to New Langfuse Org/Project

This guide walks you through uploading all prompts from `prompts.md` to a new Langfuse organization/project.

## Prerequisites

- Python 3.8+
- Langfuse Python SDK installed (`pip install langfuse`)
- Access to create a new Langfuse organization/project

## Step 1: Create New Langfuse Org/Project

1. **Go to Langfuse:**
   - Cloud: https://cloud.langfuse.com
   - Or your self-hosted instance

2. **Create Organization/Project:**
   - Create a new organization (if needed)
   - Create a new project within that organization

3. **Get API Credentials:**
   - Navigate to Project Settings → API Keys
   - Create a new API key
   - Copy the **Public Key** and **Secret Key**

## Step 2: Configure Environment Variables

Create a new `.env` file or update your existing one with the new credentials:

```bash
# For the NEW Langfuse org/project
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx
LANGFUSE_BASE_URL=https://cloud.langfuse.com  # or your self-hosted URL (e.g., http://localhost:3000)
```

**Important:** If you want to keep your existing Langfuse credentials, you can:
- Option A: Temporarily replace them (remember to backup!)
- Option B: Use a separate `.env.new` file and load it specifically

### Option B: Using a separate .env file

Create `.env.new`:
```bash
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx  # NEW org/project
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx  # NEW org/project
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Then run the upload script with:
```bash
# Load the new credentials
export $(cat .env.new | xargs)

# Run the upload
python upload_prompts_to_langfuse.py
```

## Step 3: Run the Upload Script

### Dry Run First (Recommended)

It's recommended to do a dry run first to see what will be uploaded:

```bash
python upload_prompts_to_langfuse.py --dry-run
```

This will show you all prompts that would be uploaded without actually uploading them.

### Actual Upload

```bash
python upload_prompts_to_langfuse.py
```

The script will:
1. ✅ Check your Langfuse credentials
2. 📖 Parse all prompts from `prompts.md`
3. 📋 Show you a list of prompts to be uploaded
4. ❓ Ask for confirmation
5. 🚀 Upload all prompts to Langfuse as **chat prompts** (type="chat")
6. ✅ Show a summary of results

### Skip Confirmation

To skip the confirmation prompt:

```bash
python upload_prompts_to_langfuse.py --yes
```

## What Gets Uploaded?

For each prompt in `prompts.md`, the script uploads:
- **Name:** The prompt identifier (e.g., `newsagent/battle_prompt`)
- **Type:** Chat prompt (`type="chat"`)
- **Version:** The version number
- **Labels:** Tags like `production`, `latest`
- **Config:** Model settings (model, reasoning_effort, etc.)
- **Content:** System and user prompt messages as chat records

### Chat Prompt Format

Prompts are uploaded as **chat prompts** with the following structure:
```python
[
  {"role": "system", "content": "Your system prompt here..."},
  {"role": "user", "content": "Your user prompt here..."}
]
```

This ensures both system and user messages are properly stored and retrievable as structured chat messages in Langfuse.

## Verification

After upload, verify in Langfuse:
1. Go to your project in Langfuse
2. Navigate to **Prompts** section
3. You should see all prompts from `prompts.md`
4. Click on any prompt to verify:
   - Type shows as **"Chat"**
   - Both system and user messages are visible
   - Messages are structured as chat records with roles

### Testing Prompt Retrieval

Test that prompts can be retrieved correctly:

```python
from langfuse import Langfuse

client = Langfuse()

# Get a chat prompt
prompt = client.get_prompt("newsagent/battle_prompt", type="chat")

# Compile with variables (if any)
compiled = prompt.compile()  # Returns list of chat messages

# Verify structure
print(compiled)
# Expected output:
# [
#   {"role": "system", "content": "..."},
#   {"role": "user", "content": "..."}
# ]
```

## Troubleshooting

### "Missing required Langfuse credentials"
- Ensure your `.env` file has `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`
- Check that you've loaded the environment variables

### "Connection failed"
- Verify your `LANGFUSE_BASE_URL` is correct
- Check your network connection
- Verify your API keys are valid

### "No prompts found to upload"
- Ensure `prompts.md` exists in the current directory
- Check the file format matches the expected structure

### Upload failures for specific prompts
- Check the error message for details
- Some prompts might have parsing issues in `prompts.md`
- You can manually review and fix those prompts

### Only user prompt appears in Langfuse (system prompt missing)
- This was fixed by adding `type="chat"` to the upload script
- If you uploaded prompts before this fix, re-run the upload script
- The script will create new versions with both system and user prompts properly stored
- Verify in Langfuse that prompts show as "Chat" type with both messages

## Advanced: Selective Upload

If you want to upload only specific prompts, you can modify the script or create a filtered version:

```python
# Example: Upload only prompts starting with "newsagent/"
prompts = [p for p in parse_prompts_md() if p['name'].startswith('newsagent/')]
```

## Managing Multiple Environments

If you work with multiple Langfuse environments (dev, staging, production), consider:

1. **Separate .env files:**
   - `.env.dev`
   - `.env.staging`
   - `.env.prod`

2. **Use a wrapper script:**
   ```bash
   #!/bin/bash
   # upload_to_env.sh
   ENV=${1:-dev}
   export $(cat .env.$ENV | xargs)
   python upload_prompts_to_langfuse.py
   ```

   Usage:
   ```bash
   ./upload_to_env.sh dev      # Upload to dev
   ./upload_to_env.sh staging  # Upload to staging
   ./upload_to_env.sh prod     # Upload to production
   ```

## Next Steps

After uploading:
1. ✅ Verify all prompts in Langfuse UI (check they're "Chat" type)
2. ✅ Verify both system and user messages are visible in each prompt
3. 🔄 Update your code to point to the new Langfuse project (if needed)
4. 🧪 Test prompt retrieval with `client.get_prompt("newsagent/battle_prompt", type="chat")`
5. 📝 Update any documentation with new Langfuse org/project details

**Note:** Always specify `type="chat"` when retrieving chat prompts to ensure proper type inference.

## Related Files

- `upload_prompts_to_langfuse.py` - Upload script
- `list_langfuse_prompts.py` - List and download prompts from Langfuse
- `prompts.md` - Source of truth for all prompts
- `config.py` - Langfuse client configuration
- `LANGFUSE_BACKUP_GUIDE.md` - Guide for backing up and restoring your Langfuse Docker instance
