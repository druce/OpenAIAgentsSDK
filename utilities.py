#!/usr/bin/env python3
"""
Utilities for agent workflows including status tracking and common data structures.

This module provides reusable components for managing agent workflows,
status tracking, and common data processing utilities.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

class StepStatus(Enum):
    """Status of individual workflow steps."""
    NOT_STARTED = "not_started"
    STARTED = "started"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class WorkflowStatus:
    """Tracks the status of all workflow steps."""
    step_01_gather_urls: StepStatus = StepStatus.NOT_STARTED
    step_02_filter_urls: StepStatus = StepStatus.NOT_STARTED
    step_03_download_articles: StepStatus = StepStatus.NOT_STARTED
    step_04_extract_summaries: StepStatus = StepStatus.NOT_STARTED
    step_05_cluster_by_topic: StepStatus = StepStatus.NOT_STARTED
    step_06_rate_articles: StepStatus = StepStatus.NOT_STARTED
    step_07_select_sections: StepStatus = StepStatus.NOT_STARTED
    step_08_draft_sections: StepStatus = StepStatus.NOT_STARTED
    step_09_finalize_newsletter: StepStatus = StepStatus.NOT_STARTED

    def _validate_step_name(self, step_name: str) -> None:
        """Validate that step_name is a valid step attribute."""
        if not (hasattr(self, step_name) and
                step_name.startswith('step_') and
                isinstance(getattr(self, step_name), StepStatus)):
            raise ValueError(f"Unknown step: {step_name}")

    def start_step(self, step_name: str) -> None:
        """
        Mark a step as started.

        Args:
            step_name: Name of the step attribute (e.g., "step_01_gather_urls")
        """
        self._validate_step_name(step_name)
        setattr(self, step_name, StepStatus.STARTED)

    def complete_step(self, step_name: str) -> None:
        """
        Mark a step as complete.

        Args:
            step_name: Name of the step attribute (e.g., "step_01_gather_urls")
        """
        self._validate_step_name(step_name)
        setattr(self, step_name, StepStatus.COMPLETE)

    def error_step(self, step_name: str, error_message: str = "Unknown error") -> None:
        """
        Mark a step as having an error.

        Args:
            step_name: Name of the step attribute (e.g., "step_01_gather_urls")
            error_message: Description of the error (currently not stored)
        """
        self._validate_step_name(step_name)
        setattr(self, step_name, StepStatus.ERROR)

    def get_status_dict(self) -> Dict[str, str]:
        """Get status as a dictionary for easy viewing."""
        status_dict = {}
        for attr_name in dir(self):
            if attr_name.startswith('step_'):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, StepStatus):
                    status_dict[attr_name] = attr_value.value
        return status_dict

    def all_complete(self) -> bool:
        """Check if all steps are complete."""
        status_dict = self.get_status_dict()
        return all(status == StepStatus.COMPLETE.value for status in status_dict.values())

    def get_completed_steps(self) -> List[str]:
        """Get list of completed step names."""
        status_dict = self.get_status_dict()
        return [step for step, status in status_dict.items() if status == StepStatus.COMPLETE.value]

    def get_failed_steps(self) -> List[str]:
        """Get list of failed step names."""
        status_dict = self.get_status_dict()
        return [step for step, status in status_dict.items() if status == StepStatus.ERROR.value]

    def get_started_steps(self) -> List[str]:
        """Get list of started (in-progress) step names."""
        status_dict = self.get_status_dict()
        return [step for step, status in status_dict.items() if status == StepStatus.STARTED.value]

    def get_step_status(self, step_name: str) -> StepStatus:
        """Get the status of a specific step."""
        self._validate_step_name(step_name)
        return getattr(self, step_name)

    def is_step_complete(self, step_name: str) -> bool:
        """Check if a specific step is complete."""
        return self.get_step_status(step_name) == StepStatus.COMPLETE

    def has_errors(self) -> bool:
        """Check if any steps have errors."""
        return len(self.get_failed_steps()) > 0

    def get_progress_percentage(self) -> float:
        """Get workflow completion percentage."""
        status_dict = self.get_status_dict()
        total_steps = len(status_dict)
        completed_count = len(self.get_completed_steps())
        return (completed_count / total_steps) * 100 if total_steps > 0 else 0

    def get_current_step(self) -> Optional[str]:
        """Get the current step that should be executed next."""
        status_dict = self.get_status_dict()
        # Sort step names to ensure proper order (step_01, step_02, etc.)
        sorted_steps = sorted(status_dict.keys())

        for step_name in sorted_steps:
            if status_dict[step_name] == StepStatus.NOT_STARTED.value:
                return step_name

        return None  # All steps are complete or started

    def get_status_summary(self) -> Dict[str, int]:
        """Get a summary count of steps by status."""
        status_dict = self.get_status_dict()
        summary = {status.value: 0 for status in StepStatus}

        for status in status_dict.values():
            summary[status] += 1

        return summary

    def reset(self):
        """Reset all steps to NOT_STARTED."""
        for attr_name in dir(self):
            if attr_name.startswith('step_'):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, StepStatus):
                    setattr(self, attr_name, StepStatus.NOT_STARTED)

    @staticmethod
    def format_step_name(step_name: str) -> str:
        """Format a step name for display (e.g., 'step_01_gather_urls' -> 'Step 1: Gather URLs')."""
        # Remove 'step_' prefix and split on underscore
        if not step_name.startswith('step_'):
            return step_name.replace('_', ' ').title()

        parts = step_name.replace('step_', '').split('_')

        if not parts or not parts[0].isdigit():
            return step_name.replace('_', ' ').title()

        # Remove leading zeros from step number
        step_num = parts[0].lstrip('0') or '0'
        action = ' '.join(parts[1:]).replace('_', ' ').title()

        return f"Step {step_num}: {action}"


    @staticmethod
    def get_step_icon(status: StepStatus) -> str:
        """Get emoji icon for a step status."""
        icons = {
            StepStatus.NOT_STARTED: "⭕",
            StepStatus.STARTED: "🔄",
            StepStatus.COMPLETE: "✅",
            StepStatus.ERROR: "❌"
        }
        return icons.get(status, "❓")


def create_news_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame with proper columns for news headlines."""
    return pd.DataFrame(columns=[
        'headline', 'url', 'source', 'timestamp', 'ai_related', 'cluster_topic'
    ])


def get_workflow_status_report(workflow_status: WorkflowStatus, title: str = "Workflow Status") -> str:
    """
    Generate a structured workflow status report optimized for LLM consumption.

    Returns a clean, text-based report without emojis or visual formatting.
    """
    status_dict = workflow_status.get_status_dict()
    total_steps = len(status_dict)
    completed = len(workflow_status.get_completed_steps())
    failed = len(workflow_status.get_failed_steps())
    started = len(workflow_status.get_started_steps())
    progress = workflow_status.get_progress_percentage()

    lines = [
        f"{title.upper()}",
        f"Progress: {progress:.1f}% ({completed}/{total_steps} complete)",
        f"Status Summary: {completed} complete, {started} started, {failed} failed, {total_steps - completed - started - failed} not started"
    ]

    current_step = workflow_status.get_current_step()
    if current_step:
        formatted_current = WorkflowStatus.format_step_name(current_step)
        lines.append(f"Next Step: {formatted_current}")
    else:
        lines.append("Status: All steps complete or started")

    lines.append("\nStep Details:")
    sorted_steps = sorted(status_dict.keys())
    for step_name in sorted_steps:
        status_value = status_dict[step_name]
        formatted_name = WorkflowStatus.format_step_name(step_name)
        lines.append(f"  {formatted_name}: {status_value}")

    return "\n".join(lines)


def print_workflow_summary(workflow_status: WorkflowStatus, title: str = "Workflow Status"):
    """Print a formatted summary of workflow status."""
    print(f"\n{'='*60}")
    print(f"📊 {title.upper()}")
    print('='*60)

    status_dict = workflow_status.get_status_dict()
    total_steps = len(status_dict)
    progress = workflow_status.get_progress_percentage()
    completed = len(workflow_status.get_completed_steps())

    print(f"📈 Progress: {progress:.1f}% ({completed}/{total_steps} complete)")

    current_step = workflow_status.get_current_step()
    if current_step:
        formatted_current = WorkflowStatus.format_step_name(current_step)
        print(f"➡️  Next step: {formatted_current}")
    else:
        print("🎉 All steps complete!")

    print("\n📋 Step Details:")

    sorted_steps = sorted(status_dict.keys())
    for step_name in sorted_steps:
        status_value = status_dict[step_name]
        status = StepStatus(status_value)
        icon = WorkflowStatus.get_step_icon(status)
        formatted_name = WorkflowStatus.format_step_name(step_name)
        print(f"  {icon} {formatted_name:<25} | {status.value}")

    print('='*60 + '\n')


if __name__ == "__main__":
    # Demo usage
    print("🧪 Testing WorkflowStatus utilities...")

    # Create and test workflow status
    workflow = WorkflowStatus()

    print("Initial status:")
    print_workflow_summary(workflow, "Initial Workflow")

    # Simulate some progress
    workflow.step_01_gather_urls = StepStatus.COMPLETE
    workflow.step_02_filter_urls = StepStatus.STARTED

    print("After some progress:")
    print_workflow_summary(workflow, "Updated Workflow")

    # Test utility functions
    print(f"📊 Progress: {workflow.get_progress_percentage():.1f}%")
    print(f"✅ Completed steps: {workflow.get_completed_steps()}")
    print(f"➡️  Current step: {workflow.get_current_step()}")
    print(f"🏁 All complete: {workflow.all_complete()}")

    # Test DataFrame creation
    df = create_news_dataframe()
    print(f"📰 Created DataFrame with columns: {list(df.columns)}")

    print("✅ WorkflowStatus utilities test completed!")