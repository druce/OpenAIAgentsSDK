#!/usr/bin/env python3
"""
Debug Tools - Test individual tools without the full agent
"""

import asyncio
import os
import json
import dotenv
from mock_agent import (NewsletterAgentState, GatherUrlsTool, FilterUrlsTool, 
                       DownloadArticlesTool, ExtractSummariesTool, gather_urls)
from agents import set_default_openai_client
from openai import AsyncOpenAI
from utilities import WorkflowStatus
from direct_state_access import DirectStateAccess

class MockContext:
    """Mock context object for testing tools"""
    def __init__(self, state: NewsletterAgentState):
        self.context = state

async def test_gather_urls_only():
    """Test just the URL gathering function"""
    print("🔄 Testing gather_urls function directly")
    print("=" * 40)
    
    results = await gather_urls("sources.yaml", max_concurrent=3)
    
    print(f"Total sources processed: {len(results)}")
    
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] != 'success']
    
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    
    total_articles = 0
    for result in successful_results:
        articles = result.get('results', [])
        total_articles += len(articles)
        print(f"  {result['source_key']}: {len(articles)} articles")
    
    print(f"\nTotal articles gathered: {total_articles}")
    
    if failed_results:
        print(f"\nFailed sources:")
        for result in failed_results:
            print(f"  {result['source_key']}: {result.get('metadata', {}).get('error', 'Unknown error')}")
    
    return results

async def test_tool_in_isolation(tool_class, state: NewsletterAgentState, verbose: bool = True):
    """Test a specific tool with a given state"""
    print(f"🔧 Testing {tool_class.__name__} in isolation")
    print("-" * 50)
    
    workflow_status = WorkflowStatus()
    tool = tool_class(workflow_status, verbose)
    
    # Create mock context
    mock_ctx = MockContext(state)
    
    # Get the tool's async function
    if hasattr(tool, '_gather_urls'):
        result = await tool._gather_urls(mock_ctx, "")
    elif hasattr(tool, '_filter_urls'):
        result = await tool._filter_urls(mock_ctx, "")
    elif hasattr(tool, '_download_articles'):
        result = await tool._download_articles(mock_ctx, "")
    elif hasattr(tool, '_extract_summaries'):
        result = await tool._extract_summaries(mock_ctx, "")
    else:
        result = "Tool method not found"
    
    print(f"Tool Result:\n{result}")
    print(f"\nState after tool execution:")
    print(f"  Current step: {state.current_step}")
    print(f"  Articles: {len(state.headline_data)}")
    print(f"  AI-related: {sum(1 for a in state.headline_data if a.get('ai_related') is True)}")
    
    return state

async def debug_step_by_step():
    """Debug workflow step by step with detailed logging"""
    print("🐛 STEP-BY-STEP DEBUGGING")
    print("=" * 50)
    
    # Create initial state
    state = NewsletterAgentState()
    
    print("🔄 Step 1: Gather URLs")
    print("-" * 20)
    state = await test_tool_in_isolation(GatherUrlsTool, state)
    
    print("\n🔄 Step 2: Filter URLs")  
    print("-" * 20)
    state = await test_tool_in_isolation(FilterUrlsTool, state)
    
    print("\n🔄 Step 3: Download Articles")
    print("-" * 20)
    state = await test_tool_in_isolation(DownloadArticlesTool, state)
    
    print("\n🔄 Step 4: Extract Summaries")
    print("-" * 20)
    state = await test_tool_in_isolation(ExtractSummariesTool, state)
    
    # Save final state for inspection
    accessor = DirectStateAccess("debug_step_by_step")
    accessor.save_state(state)
    
    print(f"\n✅ Debug complete. State saved as 'debug_step_by_step'")
    print(f"Use: python direct_state_access.py debug_step_by_step summary")

async def test_with_existing_state(session_id: str, tool_name: str):
    """Test a tool with an existing state"""
    print(f"🔧 Testing {tool_name} with existing state: {session_id}")
    print("-" * 50)
    
    # Load existing state
    accessor = DirectStateAccess(session_id)
    state = accessor.load_state()
    
    print(f"Loaded state - Step: {state.current_step}, Articles: {len(state.headline_data)}")
    
    # Map tool names to classes
    tool_map = {
        'gather': GatherUrlsTool,
        'filter': FilterUrlsTool, 
        'download': DownloadArticlesTool,
        'summarize': ExtractSummariesTool
    }
    
    if tool_name in tool_map:
        state = await test_tool_in_isolation(tool_map[tool_name], state)
        
        # Save modified state
        accessor.save_state(state)
        print(f"\nModified state saved back to {session_id}")
    else:
        print(f"Unknown tool: {tool_name}")
        print(f"Available: {', '.join(tool_map.keys())}")

async def main():
    # Load environment
    dotenv.load_dotenv()
    
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "gather_only":
            await test_gather_urls_only()
            
        elif command == "step_by_step":
            await debug_step_by_step()
            
        elif command == "test_tool" and len(sys.argv) >= 4:
            session_id = sys.argv[2]
            tool_name = sys.argv[3]
            await test_with_existing_state(session_id, tool_name)
            
    else:
        print("Usage:")
        print("  python debug_tools.py gather_only")
        print("  python debug_tools.py step_by_step") 
        print("  python debug_tools.py test_tool <session_id> <tool_name>")
        print("")
        print("Tool names: gather, filter, download, summarize")

if __name__ == "__main__":
    asyncio.run(main())