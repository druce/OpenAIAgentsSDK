#!/usr/bin/env python3
"""
State Inspector - Load and examine persistent agent state
"""

import asyncio
import os
import json
import dotenv
from mock_agent import NewsletterAgent, NewsletterAgentState
from agents import set_default_openai_client, SQLiteSession
from openai import AsyncOpenAI

async def inspect_state(session_id: str = "test_newsletter"):
    """Load and inspect persistent state"""
    print(f"🔍 Inspecting State for Session: {session_id}")
    print("=" * 50)

    # Load environment variables
    dotenv.load_dotenv()

    # Set up OpenAI client (needed for agent initialization)
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    default_headers_str = os.getenv("OPENAI_DEFAULT_HEADERS")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    default_headers = {}
    if default_headers_str:
        try:
            default_headers = json.loads(default_headers_str)
        except json.JSONDecodeError:
            pass

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url or "https://api.openai.com/v1",
        default_headers=default_headers
    )

    set_default_openai_client(client)

    # Load the agent with existing session
    agent = NewsletterAgent(session_id=session_id, verbose=True)
    
    # Get state by running inspection tool
    result = await agent.run_step("Just inspect the current state - don't run any workflow steps")
    
    print("🔍 State Inspection Result:")
    print(result)
    
    return agent

async def test_tool_on_state(session_id: str, tool_test: str):
    """Test a specific tool with the current state"""
    print(f"\n🧪 Testing Tool on State: {session_id}")
    print("-" * 40)
    
    agent = await inspect_state(session_id)
    
    print(f"\n🔨 Running Tool Test: {tool_test}")
    result = await agent.run_step(tool_test)
    print(f"Tool Result:\n{result}")

async def debug_workflow():
    """Debug workflow step by step"""
    print("🐛 Debugging Workflow")
    print("=" * 30)
    
    # Load environment
    dotenv.load_dotenv()
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    default_headers_str = os.getenv("OPENAI_DEFAULT_HEADERS")

    default_headers = {}
    if default_headers_str:
        try:
            default_headers = json.loads(default_headers_str)
        except:
            pass

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url or "https://api.openai.com/v1",
        default_headers=default_headers
    )
    set_default_openai_client(client)

    # Create fresh agent for debugging
    debug_session = "debug_workflow"
    agent = NewsletterAgent(session_id=debug_session, verbose=True)
    
    print("\n1️⃣ Initial State Check")
    result = await agent.run_step("Check workflow status and inspect state")
    print(f"Initial: {result[:200]}...\n")
    
    print("\n2️⃣ Run Step 1 Only")
    result = await agent.run_step("Run step 1 (gather URLs) only. Do not proceed to other steps.")
    print(f"Step 1: {result[:200]}...\n")
    
    print("\n3️⃣ Inspect State After Step 1")
    result = await agent.run_step("Inspect state and show me details about headline_data")
    print(f"Post Step 1 State:\n{result}\n")
    
    return agent

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "inspect":
            session_id = sys.argv[2] if len(sys.argv) > 2 else "test_newsletter"
            asyncio.run(inspect_state(session_id))
            
        elif command == "test_tool":
            session_id = sys.argv[2] if len(sys.argv) > 2 else "test_newsletter"
            tool_test = sys.argv[3] if len(sys.argv) > 3 else "Run step 2 (filter URLs) only"
            asyncio.run(test_tool_on_state(session_id, tool_test))
            
        elif command == "debug":
            asyncio.run(debug_workflow())
            
    else:
        print("Usage:")
        print("  python inspect_state.py inspect [session_id]")
        print("  python inspect_state.py test_tool [session_id] [tool_command]")
        print("  python inspect_state.py debug")
        print("")
        print("Examples:")
        print("  python inspect_state.py inspect simple_test")
        print("  python inspect_state.py test_tool simple_test 'Run step 2 only'")
        print("  python inspect_state.py debug")