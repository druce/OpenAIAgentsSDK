#!/usr/bin/env python3
"""
OpenAI Agents SDK tool for workflow status management.

This module provides tools for agents to check and manage workflow status
in a structured, LLM-friendly format.
"""

from typing import Dict, Any, Optional
from utilities import (
    StepStatus,
    WorkflowStatus,
    get_workflow_status_report,
    print_workflow_summary,
    format_step_name,
    get_step_icon
)
from newsletter_state import NewsletterAgentState


def get_workflow_status_tool(workflow_status: WorkflowStatus) -> Dict[str, Any]:
    """
    OpenAI Agents SDK tool definition for getting workflow status.

    Args:
        workflow_status: The WorkflowStatus instance to monitor

    Returns:
        Tool definition dict for OpenAI Agents SDK
    """

    def get_status() -> str:
        """Get current workflow status in LLM-friendly format."""
        return get_workflow_status_report(workflow_status, "Newsletter Workflow Status")

    return {
        "type": "function",
        "function": {
            "name": "get_workflow_status",
            "description": "Get the current status of the newsletter workflow including progress, next steps, and detailed step status",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        "implementation": get_status
    }


def manage_workflow_step_tool(workflow_status: WorkflowStatus) -> Dict[str, Any]:
    """
    OpenAI Agents SDK tool definition for managing workflow steps.

    Args:
        workflow_status: The WorkflowStatus instance to manage

    Returns:
        Tool definition dict for OpenAI Agents SDK
    """

    def manage_step(step_name: str, action: str, error_message: Optional[str] = None) -> str:
        """
        Manage a workflow step status.

        Args:
            step_name: Name of the step (e.g., "step_01_gather_urls")
            action: Action to perform ("start", "complete", "error")
            error_message: Error description if action is "error"

        Returns:
            Status message confirming the action
        """
        try:
            if action == "start":
                workflow_status.start_step(step_name)
                formatted_name = WorkflowStatus.format_step_name(step_name)
                return f"Started: {formatted_name}"
            elif action == "complete":
                workflow_status.complete_step(step_name)
                formatted_name = WorkflowStatus.format_step_name(step_name)
                return f"Completed: {formatted_name}"
            elif action == "error":
                workflow_status.error_step(step_name, error_message or "Unknown error")
                formatted_name = WorkflowStatus.format_step_name(step_name)
                return f"Marked as error: {formatted_name} - {error_message or 'Unknown error'}"
            else:
                return f"Invalid action: {action}. Use 'start', 'complete', or 'error'"
        except ValueError as e:
            return f"Error: {str(e)}"

    return {
        "type": "function",
        "function": {
            "name": "manage_workflow_step",
            "description": "Start, complete, or mark a workflow step as having an error",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_name": {
                        "type": "string",
                        "description": "Name of the workflow step (e.g., 'step_01_gather_urls')",
                        "enum": [
                            "step_01_gather_urls",
                            "step_02_filter_urls",
                            "step_03_download_articles",
                            "step_04_extract_summaries",
                            "step_05_rate_articles",
                            "step_06_cluster_by_topic",
                            "step_07_create_sections",
                            "step_08_draft_sections",
                            "step_09_finalize_newsletter"
                        ]
                    },
                    "action": {
                        "type": "string",
                        "description": "Action to perform on the step",
                        "enum": ["start", "complete", "error"]
                    },
                    "error_message": {
                        "type": "string",
                        "description": "Description of the error (required if action is 'error')"
                    }
                },
                "required": ["step_name", "action"]
            }
        },
        "implementation": manage_step
    }


def create_workflow_tools(workflow_status: WorkflowStatus) -> list[Dict[str, Any]]:
    """
    Create all workflow management tools for OpenAI Agents SDK.

    Args:
        workflow_status: The WorkflowStatus instance to manage

    Returns:
        List of tool definitions for OpenAI Agents SDK
    """
    return [
        get_workflow_status_tool(workflow_status),
        manage_workflow_step_tool(workflow_status)
    ]


# Example usage for OpenAI Agents SDK
if __name__ == "__main__":
    # Initialize workflow
    workflow = WorkflowStatus()

    # Create tools
    tools = create_workflow_tools(workflow)

    # Example of how to use with OpenAI Agents SDK
    print("Tool definitions created:")
    for tool in tools:
        print(f"- {tool['function']['name']}: {tool['function']['description']}")

    # Test the tools
    print("\nTesting workflow status tool:")
    status_tool = tools[0]
    print(status_tool['implementation']())

    print("\nTesting step management tool:")
    manage_tool = tools[1]
    print(manage_tool['implementation']("step_01_gather_urls", "start"))
    print(manage_tool['implementation']("step_01_gather_urls", "complete"))

    print("\nUpdated status:")
    print(status_tool['implementation']())
