#!/usr/bin/env python3
"""
Newsletter Agent State Management.

This module provides the NewsletterAgentState class and related utilities
for managing the complete newsletter generation workflow state.
"""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import pandas as pd
from enum import Enum
import json
import sqlite3
from config import DEFAULT_CONCURRENCY


class StepStatus(Enum):
    """Status of individual workflow steps."""
    NOT_STARTED = "not_started"
    STARTED = "started"
    COMPLETE = "complete"
    ERROR = "error"
    SKIPPED = "skipped"


class WorkflowStep(BaseModel):
    """Individual workflow step with tracking metadata."""
    id: str = Field(description="Unique step identifier (e.g., 'gather_urls')")
    name: str = Field(description="Human-readable step name")
    description: str = Field(description="What this step does")
    status: StepStatus = Field(default=StepStatus.NOT_STARTED)

    # Metadata for tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: str = ""
    status_message: str = ""  # Success/info message for the step
    retry_count: int = 0

    # Optional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def start(self):
        """Mark step as started."""
        self.status = StepStatus.STARTED
        self.started_at = datetime.now()

    def complete(self):
        """Mark step as complete."""
        self.status = StepStatus.COMPLETE
        self.completed_at = datetime.now()

    def error(self, message: str):
        """Mark step as error."""
        self.status = StepStatus.ERROR
        self.error_message = message
        self.retry_count += 1

    def __str__(self) -> str:
        """Return a nice string representation of the workflow step."""
        # Format timing information if available
        timing = ""
        if self.started_at and self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds()
            timing = f" (took {duration:.1f}s)"
        elif self.started_at:
            duration = (datetime.now() - self.started_at).total_seconds()
            timing = f" (running {duration:.1f}s)"

        # Build status indicator
        status_emoji = {
            StepStatus.NOT_STARTED: "⚪",
            StepStatus.STARTED: "🔵",
            StepStatus.COMPLETE: "✅",
            StepStatus.ERROR: "❌",
            StepStatus.SKIPPED: "⏭️"
        }
        emoji = status_emoji.get(self.status, "❓")

        # Base representation
        result = f"{emoji} {self.name} [{self.status.value}]{timing}"

        # Add error or status message if present
        if self.error_message:
            result += f"\n   Error: {self.error_message}"
        else:
            result += f"\n   {self.status_message}"

        return result

    def __repr__(self) -> str:
        """Return simple representation for debugging."""
        return f"WorkflowStep(id='{self.id}', name='{self.name}', status={self.status.value}, retry_count={self.retry_count})"


class WorkflowState(BaseModel):
    """
    Generic workflow state base class - can be subclassed for specific agents.

    Provides flexible workflow tracking using a list of steps instead of
    hardcoded step_XX fields, making it easy to insert/remove steps.
    """
    steps: List[WorkflowStep] = Field(default_factory=list)
    current_step_name: str = Field(
        default="", description="name of currently executing step")

    def add_step(self, step_id: str, name: str, description: str) -> 'WorkflowState':
        """
        Fluent API for building workflows.

        Args:
            step_id: Unique identifier for the step
            name: Human-readable name
            description: What the step does

        Returns:
            Self for method chaining
        """
        self.steps.append(WorkflowStep(
            id=step_id,
            name=name,
            description=description
        ))
        return self

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_step_index(self, step_id: str) -> int:
        """Get the index of a step in the workflow (0-based)."""
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                return i
        return -1

    def start_step(self, step_id: str) -> None:
        """
        Mark a step as started.

        Args:
            step_id: ID of the step to start
        """
        step = self.get_step(step_id)
        if not step:
            raise ValueError(f"Unknown step: {step_id}")

        self.current_step_name = step_id
        step.start()

    def complete_step(self, step_id: str, message: str = "") -> None:
        """
        Mark a step as complete with optional status message.

        Args:
            step_id: ID of the step to complete
            message: Optional status message describing what was accomplished
        """
        step = self.get_step(step_id)
        if not step:
            raise ValueError(f"Unknown step: {step_id}")

        step.complete()
        if message:
            step.status_message = message

    def error_step(self, step_id: str, error_message: str = "Unknown error") -> None:
        """
        Mark a step as having an error.

        Args:
            step_id: ID of the step with error
            error_message: Description of the error
        """
        step = self.get_step(step_id)
        if not step:
            raise ValueError(f"Unknown step: {step_id}")

        step.error(error_message)

    def get_step_status(self, step_id: str) -> StepStatus:
        """Get the status of a specific step."""
        step = self.get_step(step_id)
        if not step:
            raise ValueError(f"Unknown step: {step_id}")
        return step.status

    def is_step_complete(self, step_id: str) -> bool:
        """Check if a specific step is complete."""
        return self.get_step_status(step_id) == StepStatus.COMPLETE

    def all_complete(self) -> bool:
        """Check if all steps are complete."""
        return all(s.status == StepStatus.COMPLETE for s in self.steps)

    def has_errors(self) -> bool:
        """Check if any steps have errors."""
        return any(s.status == StepStatus.ERROR for s in self.steps)

    def get_completed_steps(self) -> List[str]:
        """Get list of completed step IDs."""
        return [s.id for s in self.steps if s.status == StepStatus.COMPLETE]

    def get_failed_steps(self) -> List[str]:
        """Get list of failed step IDs."""
        return [s.id for s in self.steps if s.status == StepStatus.ERROR]

    def get_started_steps(self) -> List[str]:
        """Get list of started (in-progress) step IDs."""
        return [s.id for s in self.steps if s.status == StepStatus.STARTED]

    def get_current_step(self) -> Optional[str]:
        """Get the first step ID that's not COMPLETE (the next step to execute)."""
        for step in self.steps:
            if step.status != StepStatus.COMPLETE:
                return step.id
        return None  # All steps complete

    def get_progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if not self.steps:
            return 0.0
        completed = len(self.get_completed_steps())
        return (completed / len(self.steps)) * 100

    def get_status_summary(self) -> Dict[str, int]:
        """Get a summary count of steps by status."""
        summary = {status.value: 0 for status in StepStatus}
        for step in self.steps:
            summary[step.status.value] += 1
        return summary

    def clear_errors(self):
        """Reset all steps with ERROR status back to NOT_STARTED."""
        for step in self.steps:
            if step.status == StepStatus.ERROR:
                step.status = StepStatus.NOT_STARTED
                step.error_message = ""

    def reset(self):
        """Reset all steps to NOT_STARTED."""
        for step in self.steps:
            step.status = StepStatus.NOT_STARTED
            step.started_at = None
            step.completed_at = None
            step.error_message = ""
            step.retry_count = 0
        self.current_step_name = ""

    @staticmethod
    def format_step_name(step_id: str, step_name: str, step_index: int) -> str:
        """
        Format a step for display.

        Args:
            step_id: Step identifier
            step_name: Human-readable name
            step_index: 0-based index in workflow

        Returns:
            Formatted string like "Step 0: Gather URLs"
        """
        return f"Step {step_index}: {step_name}"

    @staticmethod
    def get_step_icon(status: StepStatus) -> str:
        """Get emoji icon for a step status."""
        icons = {
            StepStatus.NOT_STARTED: "⭕",
            StepStatus.STARTED: "🔄",
            StepStatus.COMPLETE: "✅",
            StepStatus.ERROR: "❌",
            StepStatus.SKIPPED: "⏭️"
        }
        return icons.get(status, "❓")

    def get_workflow_status_report(self, title: str = "Workflow Status") -> str:
        """
        Generate a structured workflow status report optimized for LLM consumption.

        Returns a clean, text-based report without emojis or visual formatting.
        """
        total_steps = len(self.steps)
        completed = len(self.get_completed_steps())
        failed = len(self.get_failed_steps())
        started = len(self.get_started_steps())
        progress = self.get_progress_percentage()

        lines = [
            f"{title.upper()}",
            f"Progress: {progress:.1f}% ({completed}/{total_steps} complete)",
            f"Status Summary: {completed} complete, {started} started, {failed} failed, {total_steps - completed - started - failed} not started"
        ]

        current_step = self.get_current_step()
        if current_step:
            step = self.get_step(current_step)
            step_idx = self.get_step_index(current_step)
            formatted_current = self.format_step_name(
                current_step, step.name if step else "", step_idx)
            lines.append(f"Next Step: {formatted_current}")
        else:
            lines.append("Status: All steps complete or started")

        lines.append("\nStep Details:")
        for i, step in enumerate(self.steps):
            formatted_name = self.format_step_name(step.id, step.name, i)
            lines.append(f"  {formatted_name}: {step.status.value}")
            if step.error_message:
                lines.append(f"    Error: {step.error_message}")
            if step.status_message:
                lines.append(f"    Status: {step.status_message}")

        return "\n".join(lines)

    def print_workflow_status(self, title: str = "Workflow Status"):
        """Print a human-formatted summary of workflow status."""
        print(f"\n{'='*60}")
        print(f"📊 {title.upper()}")
        print('='*60)

        total_steps = len(self.steps)
        progress = self.get_progress_percentage()
        completed = len(self.get_completed_steps())

        print(
            f"📈 Progress: {progress:.1f}% ({completed}/{total_steps} complete)")

        current_step = self.get_current_step()
        if current_step:
            step = self.get_step(current_step)
            step_idx = self.get_step_index(current_step)
            formatted_current = self.format_step_name(
                current_step, step.name if step else "", step_idx)
            print(f"➡️  Next step: {formatted_current}")
        else:
            print("🎉 All steps complete!")

        print("\n📋 Step Details:")

        for i, step in enumerate(self.steps):
            icon = self.get_step_icon(step.status)
            formatted_name = self.format_step_name(step.id, step.name, i)
            print(f"  {icon} {formatted_name:<30} | {step.status.value}")
            if step.error_message:
                print(f"      ❌ Error: {step.error_message}")
            if step.status_message:
                print(f"      📝 Status: {step.status_message}")

        print('='*60 + '\n')

    @property
    def workflow_status_message(self) -> str:
        """
        Get the most recent status message from workflow steps.
        Computed on-the-fly for backward compatibility.

        Priority:
        1. Error message from failed steps (most recent)
        2. Status message from current step
        3. Status message from last completed step

        Returns:
            Status/error message string, or empty string if none
        """
        # Priority 1: Get error message from most recent failed step
        failed_steps = self.get_failed_steps()
        if failed_steps:
            # Get the last failed step
            last_failed_step = self.get_step(failed_steps[-1])
            if last_failed_step and last_failed_step.error_message:
                return last_failed_step.error_message

        # Priority 2: Get status message from current step
        current_step_id = self.get_current_step()
        if current_step_id:
            current_step = self.get_step(current_step_id)
            if current_step:
                if current_step.status_message:
                    return current_step.status_message
                if current_step.error_message:
                    return current_step.error_message

        # Priority 3: Get status from last completed step
        completed = self.get_completed_steps()
        if completed:
            last_completed_step = self.get_step(completed[-1])
            if last_completed_step and last_completed_step.status_message:
                return last_completed_step.status_message

        return ""


class NewsletterAgentState(WorkflowState):
    """
    Persistent state for the newsletter agent workflow.

    Manages the newsletter generation process with serializable storage
    of headlines, processing of results, and workflow progress with resumable execution.

    """
    # Serializable data storage (DataFrame as list of dicts)
    headline_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of headline dictionaries with columns: title, url, source, timestamp, is_ai, summary,etc."
    )

    # Configuration
    max_edits: int = Field(default=2, description="Maximum editing iterations")
    concurrency: int = Field(
        default=DEFAULT_CONCURRENCY, description="Number of concurrent browsers")
    do_download: bool = Field(
        default=True, description="Whether to fetch sources or use already downloaded")
    reprocess_since: Optional[datetime] = Field(
        default=None, description="Only process articles since this date")

    # Source config
    sources_file: str = Field(
        default="sources.yaml", description="YAML filename containing source configurations")
    sources: Dict[str, Any] = Field(
        default_factory=dict, description="Dictionary of source configurations loaded from YAML")

    # Session management
    session_id: str = Field(
        default="", description="Unique session identifier for this workflow instance")
    db_path: str = Field(default="newsletter_agent.db",
                         description="Path to SQLite database file")

    # Topics and clustering
    cluster_names: List[str] = Field(
        default_factory=list, description="List of topic names for categorization")
    clusters: Dict[str, List[str]] = Field(
        default_factory=dict, description="Topic name -> list of article IDs and related info")
    common_topics: List[str] = Field(
        default_factory=list, description="Topics that appear in multiple articles")

    # Newsletter content
    # Note: Article summaries are now stored directly in headline_data as 'summary' column
    newsletter_section_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of newsletter section story dictionaries with columns: cat, section_title, id, headline, rating, prune, links"
    )
    newsletter_title: str = Field(
        default="", description="Generated newsletter title (H1)")
    final_newsletter: str = Field(
        default="", description="Final newsletter content")

    def __init__(self, session_id: str = "", db_path: str = "newsletter_agent.db", **kwargs):
        """Initialize NewsletterAgentState with session_id and db_path."""
        super().__init__(session_id=session_id, db_path=db_path, **kwargs)

        # Initialize workflow steps if not already set (e.g., when loading from DB)
        if not self.steps:
            self._initialize_workflow()

    def _initialize_workflow(self):
        """Define the 9 newsletter workflow steps."""
        (self
         .add_step("gather_urls", "Gather URLs", "Collect headlines and URLs from various sources")
         .add_step("filter_urls", "Filter URLs", "Filter headlines to AI-related content only")
         .add_step("download_articles", "Download Articles", "Fetch full article content from URLs")
         .add_step("extract_summaries", "Extract Summaries", "Create bullet point summaries of each article")
         .add_step("rate_articles", "Rate Articles", "Evaluate article quality and importance")
         .add_step("cluster_topics", "Cluster By Topic", "Group articles by thematic topics")
         .add_step("select_sections", "Select Sections", "Organize articles into newsletter sections")
         .add_step("draft_sections", "Draft Sections", "Write content for each section")
         .add_step("finalize_newsletter", "Finalize Newsletter", "Combine sections into final newsletter"))

    @staticmethod
    def _migrate_old_state_format(state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate state data from old format (step_XX fields) to new format (steps list).

        Args:
            state_data: Raw state dictionary loaded from database

        Returns:
            Migrated state dictionary
        """
        # Mapping from old step field names to new step IDs
        OLD_TO_NEW_STEP_MAP = {
            "step_01_fetch_urls": "gather_urls",
            "step_02_filter_urls": "filter_urls",
            "step_03_download_articles": "download_articles",
            "step_04_extract_summaries": "extract_summaries",
            "step_05_rate_articles": "rate_articles",
            "step_06_cluster_by_topic": "cluster_topics",
            "step_07_select_sections": "select_sections",
            "step_08_draft_sections": "draft_sections",
            "step_09_finalize_newsletter": "finalize_newsletter",
        }

        # Check if this is old format (has step_XX fields instead of steps list)
        has_old_format = any(key.startswith("step_0") and key in OLD_TO_NEW_STEP_MAP
                             for key in state_data.keys())

        if not has_old_format:
            # Already new format or no workflow data
            return state_data

        print("🔄 Migrating old workflow state format to new format...")

        # Create new steps list from old step fields
        steps = []
        step_definitions = [
            ("gather_urls", "Gather URLs",
             "Collect headlines and URLs from various sources"),
            ("filter_urls", "Filter URLs",
             "Filter headlines to AI-related content only"),
            ("download_articles", "Download Articles",
             "Fetch full article content from URLs"),
            ("extract_summaries", "Extract Summaries",
             "Create bullet point summaries of each article"),
            ("rate_articles", "Rate Articles",
             "Evaluate article quality and importance"),
            ("cluster_topics", "Cluster By Topic",
             "Group articles by thematic topics"),
            ("select_sections", "Select Sections",
             "Organize articles into newsletter sections"),
            ("draft_sections", "Draft Sections", "Write content for each section"),
            ("finalize_newsletter", "Finalize Newsletter",
             "Combine sections into final newsletter"),
        ]

        # Reverse mapping for status extraction
        NEW_TO_OLD_STEP_MAP = {v: k for k, v in OLD_TO_NEW_STEP_MAP.items()}

        for new_id, name, description in step_definitions:
            old_field = NEW_TO_OLD_STEP_MAP.get(new_id)
            status_value = state_data.get(old_field, "not_started")

            # Convert status value if it's stored as an enum
            if isinstance(status_value, dict) and "value" in status_value:
                status_value = status_value["value"]
            elif hasattr(status_value, "value"):
                status_value = status_value.value

            steps.append({
                "id": new_id,
                "name": name,
                "description": description,
                "status": status_value,
                "started_at": None,
                "completed_at": None,
                "error_message": "",
                "retry_count": 0,
                "metadata": {}
            })

        # Add steps to state data
        state_data["steps"] = steps

        # Migrate current_step if it exists
        if "current_step" in state_data:
            old_current = state_data.get("current_step", "")
            new_current = OLD_TO_NEW_STEP_MAP.get(old_current, "")
            state_data["current_step_id"] = new_current

        # Remove old step fields
        for old_field in OLD_TO_NEW_STEP_MAP.keys():
            state_data.pop(old_field, None)

        # Remove old workflow status fields
        state_data.pop("current_step", None)
        state_data.pop("workflow_status", None)
        state_data.pop("workflow_status_message", None)

        print(f"✅ Migration complete: converted {len(steps)} workflow steps")

        return state_data

    @property
    def newsletter_section_df(self) -> 'pd.DataFrame':
        """Newsletter section data as a DataFrame"""
        return pd.DataFrame(self.newsletter_section_data)

    # Helper methods
    @classmethod
    def create_headline_df(cls) -> pd.DataFrame:
        """Create an empty DataFrame with proper columns for news headlines."""
        return pd.DataFrame(columns=[
            'id',
            'source',
            'title',
            'orig_url',
            'url',
            'text_path',
            'site_name',
            'published',
            'is_ai',
            'topic_list',
            'cluster',
            'rating',
            'summary'
        ])

    @property
    def headline_df(self) -> 'pd.DataFrame':
        """Headline data as a DataFrame"""
        return pd.DataFrame(self.headline_data)

    def headline_df_to_dict(self, df: 'pd.DataFrame'):
        """Update headline data from DataFrame"""
        self.headline_data = df.to_dict(orient='records')

    def add_headlines(self, new_headlines: List[Dict[str, Any]]) -> None:
        """
        Add new headlines to the DataFrame with deduplication.

        Args:
            new_headlines: List of dictionaries with headline data
                          Expected keys: title, url, source, timestamp, etc.
        """
        if not new_headlines:
            print("⚠️  No new headlines to add")
            return

        new_df = pd.DataFrame(new_headlines)

        if self.headline_data:
            existing_df = self.headline_dict_to_df()
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            self.headline_df_to_dict(combined_df)
        else:
            self.headline_df_to_dict(new_df)

        print(f"📰 Added headlines - updated count: {len(self.headline_data)}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get a summary of the current session state.

        Returns:
            Dictionary with key metrics and status information
        """
        total_headlines = len(self.headline_data)

        return {
            "headlines": {
                "total": total_headlines,
                # group by source
                # ai vs non-ai
                # downloaded: have text path
                # have summary
                # have topics
                # have cluster assignment
                # have rating
            },
            "sources": {
                "config_file": self.sources_file,
                "loaded_sources": len(self.sources),
            },
            "topics": {
                "cluster_topics": len(self.cluster_names),
                "topics": self.cluster_names
            },
            "workflow": {
                "current_step": self.get_current_step(),
                "workflow_complete": self.all_complete(),
                "progress_percentage": self.get_progress_percentage(),
                "completed_steps": len(self.get_completed_steps()),
                "total_steps": len(self.steps),
                "max_edits": self.max_edits,
                "concurrency": self.concurrency,
                "do_download": self.do_download,
                "reprocess_since": self.reprocess_since
            },
            "processing": {
                "topic_clusters": len(self.clusters),
                "newsletter_sections": len(self.newsletter_section_data),
                "final_newsletter_length": len(self.final_newsletter)
            }
        }

    def print_status(self) -> None:
        """Print a formatted summary of the current session state."""
        status = self.get_status()

        print("\n" + "="*50)
        print("📊 NEWSLETTER AGENT STATE SUMMARY")
        print("="*50)

        print(f"📰 Headlines: {status['headlines']['total']} total")
        # print(f"   🤖 AI-related: {status['headlines']['is_ai']} ({status['headlines']['ai_percentage']})")
        # print(f"   📄 Non-AI: {status['headlines']['non_ai']}")

        print(f"\n📡 Sources: {status['sources']['loaded_sources']} loaded")
        print(f"   📁 Config: {status['sources']['config_file']}")

        print(
            f"\n🏷️  Topics: {status['topics']['cluster_topics']} cluster topics")
        if status['topics']['topics']:
            print(f"   📋 Topics: {', '.join(status['topics']['topics'])}")

        print("\n⚙️  Workflow:")
        print(f"   📍 Current step: {status['workflow']['current_step']}")
        print(f"   ✅ Complete: {status['workflow']['workflow_complete']}")
        print(
            f"   📊 Progress: {status['workflow']['progress_percentage']:.1f}% ({status['workflow']['completed_steps']}/{status['workflow']['total_steps']})")
        print(f"   ✏️ Max edits: {status['workflow']['max_edits']}")
        print(f"   🌐 Concurrency: {status['workflow']['concurrency']}")
        print(f"   📡 Do download: {status['workflow']['do_download']}")
        print(f"   📅 Reprocess since: {status['workflow']['reprocess_since']}")

        print("\n🔄 Processing:")
        # print(f"   📝 Article summaries: {status['processing']['article_summaries']}")
        print(
            f"   🏷️  Topic clusters: {status['processing']['topic_clusters']}")
        print(
            f"   📑 Newsletter sections: {status['processing']['newsletter_sections']}")
        if status['processing']['final_newsletter_length'] > 0:
            print(
                f"   📰 Final newsletter: {status['processing']['final_newsletter_length']} chars")

        print("="*50 + "\n")

    def get_unique_sources(self) -> List[str]:
        """Get list of unique source names from headline data."""
        df = self.headline_df
        df['count'] = 1
        df = df[['source', 'count']].groupby(['source']).sum().reset_index()
        print(df)
        return df.to_dict(orient='records')

    def serialize_to_db(self, step_name: str) -> None:
        """Serialize state to database indexed by session_id and step_name"""
        from db import AgentState
        from datetime import datetime

        with sqlite3.connect(self.db_path) as conn:
            # Create table with automatic migration
            AgentState.create_table(conn)

            # Create AgentState record
            state_record = AgentState(
                session_id=self.session_id,
                step_name=step_name,
                state_data=self.model_dump_json(),
                updated_at=datetime.now()
            )

            # Upsert the state
            state_record.upsert(conn)

    def load_from_db(self, step_name: str) -> Optional['NewsletterAgentState']:
        """Load state from database by session_id and step_name"""
        from db import AgentState

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get state record
                state_record = AgentState.get_by_session_and_step(
                    conn, self.session_id, step_name
                )

                if state_record:
                    state_data = json.loads(state_record.state_data)
                    # Apply migration if needed
                    state_data = self._migrate_old_state_format(state_data)
                    return self.__class__(**state_data)
                return None

        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return None

    def load_latest_from_db(self) -> Optional['NewsletterAgentState']:
        """Load the most recent state for a session (latest step by updated_at)"""
        from db import AgentState

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get latest state record for session
                state_record = AgentState.get_latest_by_session(
                    conn, self.session_id
                )

                if state_record:
                    state_data = json.loads(state_record.state_data)
                    # Apply migration if needed
                    state_data = self._migrate_old_state_format(state_data)
                    return self.__class__(**state_data)
                return None

        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return None

    def list_session_steps(self) -> List[Dict[str, str]]:
        """List all saved steps for a session with their timestamps"""
        from db import AgentState

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all state records for session
                state_records = AgentState.get_all_by_session(
                    conn, self.session_id
                )

                return [
                    {
                        "step_name": record.step_name,
                        "updated_at": record.updated_at.isoformat() if record.updated_at else None
                    }
                    for record in state_records
                ]

        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return []

    def save_checkpoint(self, step_name: str) -> None:
        """Save state checkpoint after completing a workflow step"""
        self.serialize_to_db(step_name)
        print(
            f"💾 State checkpoint saved for session {self.session_id} after {step_name}")

    def display_newsletter(self) -> str:
        """Display the current newsletter section data in a readable format using IPython display"""
        from IPython.display import display, Markdown

        retval = ""
        if not self.newsletter_section_data:
            print("No newsletter sections to display")
            return retval

        # Display newsletter title if available
        if self.newsletter_title:
            retval += f"# {self.newsletter_title}\n"

        # Get DataFrame sorted by sort_order and rating
        df = self.newsletter_section_df
        if 'sort_order' in df.columns:
            df = df.sort_values(['sort_order', 'rating'],
                                ascending=[True, False])
        elif 'rating' in df.columns:
            df = df.sort_values('rating', ascending=False)

        last_cat = ""
        for row in df.itertuples():
            # Display section title when category changes
            if row.cat != last_cat:
                retval += f"## {row.section_title}\n"
                last_cat = row.cat

            # Display story headline with links
            retval += f"- {row.headline} - {row.links}\n"

        return retval


if __name__ == "__main__":
    # Demo usage
    print("🚀 Creating demo newsletter state...")

    # Create state with custom topics
    state = NewsletterAgentState(
        session_id="demo_test_session", db_path="newsletter_agent.db")

    # Add some demo headlines
    demo_headlines = [
        {
            'id': 1,
            'source': 'TechCrunch',
            'title': 'AI Breakthrough in Language Models',
            'orig_url': 'https://techcrunch.com/ai-breakthrough-in-language-models',
            'url': 'https://techcrunch.com/ai-breakthrough-in-language-models',
            'text_path': 'techcrunch_ai_breakthrough_in_language_models.txt',
            'site_name': 'TechCrunch',
            'published': '2024-01-15',
            'is_ai': True,
            'topic_list': ['AI/ML', 'NLP'],
            'cluster': 'AI/ML',
            'rating': 5,
            'summary': 'New language model shows unprecedented capabilities'
        },
        {
            'id': 2,
            'source': 'venture_beat',
            "title": "Robotics Company Raises $100M",
            "orig_url": "https://example.com/robotics-funding",
            "url": "https://example.com/robotics-funding",
            'text_path': 'venture_beat_robotics_funding.txt',
            'site_name': 'venture_beat',
            'published': '2024-01-15',
            'is_ai': True,
            'topic_list': ['AI/ML', 'NLP'],
            'cluster': 'AI/ML',
            'rating': 5,
            "summary": "Major funding round for autonomous robotics"
        },
    ]

    state.add_headlines(demo_headlines)
    state.print_status()

    print("Count of unique sources:")
    print("="*50 + "\n")
    state.get_unique_sources()
    print("="*50 + "\n")

    # Test DataFrame creation
    empty_df = NewsletterAgentState.create_headline_df()
    print(f"📊 Created empty DataFrame with columns: {list(empty_df.columns)}")

    # Test state serialization
    print("\n🧪 Testing state serialization...")

    # Save state for multiple steps
    state.serialize_to_db("gather_urls")
    print(
        f"💾 Saved state for session: {state.session_id}, step: gather_urls")

    state.serialize_to_db("filter_urls")
    print(
        f"💾 Saved state for session: {state.session_id}, step: filter_urls")

    # Load specific step state
    loaded_state = state.load_from_db("gather_urls")
    if loaded_state:
        print(
            f"✅ Successfully loaded state for gather_urls with {len(loaded_state.headline_data)} articles")
        print(f"📊 Progress: {loaded_state.get_progress_percentage():.1f}%")
        print(f"📊 Current step: {loaded_state.get_current_step()}")
    else:
        print("❌ Failed to load state for gather_urls")

    # Load latest state
    latest_state = state.load_latest_from_db()
    if latest_state:
        print(
            f"✅ Successfully loaded latest state with {len(latest_state.headline_data)} articles")
    else:
        print("❌ Failed to load latest state")

    # List all steps for session
    steps = state.list_session_steps()
    print(
        f"📋 Steps saved for session {state.session_id}: {[step['step_name'] for step in steps]}")

    # Test checkpoint save
    state.save_checkpoint("test_step")

    print("✅ Newsletter state demo completed!")
