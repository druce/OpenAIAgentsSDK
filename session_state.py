#!/usr/bin/env python3
"""
Session state management for OpenAI Agents SDK workflows.

This module provides a dataclass for maintaining state across agent runs,
including data processing, source management, and workflow control variables.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import yaml


@dataclass
class AgentSessionState:
    """
    Maintains session state for OpenAI Agents SDK workflows.

    This dataclass holds all the variables needed to track progress through
    a multi-step agent workflow, including data processing, source management,
    and iteration control.

    Attributes:
        headline_df: DataFrame containing headline data for processing
        sources_file: Path to YAML file containing source configurations
        sources: Dictionary of source configurations loaded from YAML
        cluster_topics: List of clean topic names for headline categorization
        max_edits: Maximum number of critic optimizer editing iterations allowed
        edit_complete: Boolean flag indicating if editing process is finished
        n_browsers: Number of concurrent Playwright browser instances for downloads
    """

    # Core data storage
    headline_df: pd.DataFrame = field(
        default_factory=pd.DataFrame,
        metadata={
            "description": "DataFrame containing headlines to be processed",
            "columns_expected": ["headline", "url", "source", "timestamp", "ai_related"]
        }
    )

    # Source management
    sources_file: str = field(
        default="sources.yaml",
        metadata={
            "description": "YAML filename containing source configurations",
            "example": "sources.yaml"
        }
    )

    sources: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Dictionary of source configurations loaded from YAML",
            "structure": "source_name -> {url, title, rss_feed, etc.}"
        }
    )

    # Topic clustering
    cluster_topics: List[str] = field(
        default_factory=list,
        metadata={
            "description": "List of clean topic names for headline categorization",
            "example": ["AI/ML", "Robotics", "NLP", "Computer Vision", "Other"]
        }
    )

    # Workflow control
    max_edits: int = field(
        default=3,
        metadata={
            "description": "Maximum number of critic optimizer editing iterations",
            "range": "1-10 recommended"
        }
    )

    edit_complete: bool = field(
        default=False,
        metadata={
            "description": "Flag indicating if the editing process is complete",
            "usage": "Set to True when critic optimizer finishes or max_edits reached"
        }
    )

    # Browser/download management
    n_browsers: int = field(
        default=3,
        metadata={
            "description": "Number of concurrent Playwright browser instances",
            "range": "1-10 recommended, depends on system resources"
        }
    )

    def __post_init__(self):
        """
        Post-initialization validation and setup.

        Validates that the configuration makes sense and performs
        any necessary initialization steps.
        """
        # Validate max_edits is reasonable
        if self.max_edits < 1 or self.max_edits > 20:
            raise ValueError(f"max_edits should be between 1-20, got {self.max_edits}")

        # Validate n_browsers is reasonable
        if self.n_browsers < 1 or self.n_browsers > 50:
            raise ValueError(f"n_browsers should be between 1-50, got {self.n_browsers}")

        # Validate sources_file exists if it's not the default
        if self.sources_file != "sources.yaml":
            sources_path = Path(self.sources_file)
            if not sources_path.exists():
                raise FileNotFoundError(f"Sources file not found: {self.sources_file}")

    def load_sources(self) -> None:
        """
        Load source configurations from the YAML file.

        Reads the sources_file and populates the sources dictionary.
        Creates a reverse mapping from filenames to source names for convenience.

        Raises:
            FileNotFoundError: If sources_file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        try:
            sources_path = Path(self.sources_file)
            with open(sources_path, 'r', encoding='utf-8') as file:
                self.sources = yaml.safe_load(file) or {}

            print(f"✅ Loaded {len(self.sources)} sources from {self.sources_file}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Sources file not found: {self.sources_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {self.sources_file}: {e}")

    def get_sources_reverse_map(self) -> Dict[str, str]:
        """
        Create reverse mapping from filenames to source names.

        Useful for mapping downloaded HTML files back to their source configurations.

        Returns:
            Dict mapping filename -> source_name

        Example:
            {"techcrunch.html": "techcrunch", "ars_technica.html": "ars_technica"}
        """
        if not self.sources:
            self.load_sources()

        return {
            f"{source_config.get('title', source_name)}.html": source_name
            for source_name, source_config in self.sources.items()
            if isinstance(source_config, dict)
        }

    def reset_edit_state(self) -> None:
        """
        Reset the editing state for a new iteration.

        Sets edit_complete to False, useful when starting a new
        optimization cycle.
        """
        self.edit_complete = False
        print("🔄 Edit state reset - ready for new optimization cycle")

    def increment_edit_count(self, current_count: int) -> bool:
        """
        Check if we should continue editing based on current count.

        Args:
            current_count: Current number of edit iterations completed

        Returns:
            True if should continue editing, False if max reached
        """
        if current_count >= self.max_edits:
            self.edit_complete = True
            print(f"🏁 Maximum edits ({self.max_edits}) reached - marking complete")
            return False

        print(f"📝 Edit iteration {current_count + 1}/{self.max_edits}")
        return True

    def add_headlines_to_df(self, new_headlines: List[Dict[str, Any]]) -> None:
        """
        Add new headlines to the DataFrame.

        Args:
            new_headlines: List of dictionaries with headline data
                          Expected keys: headline, url, source, timestamp, etc.
        """
        if not new_headlines:
            print("⚠️  No new headlines to add")
            return

        new_df = pd.DataFrame(new_headlines)

        if self.headline_df.empty:
            self.headline_df = new_df
        else:
            # Concatenate and remove duplicates based on URL
            combined_df = pd.concat([self.headline_df, new_df], ignore_index=True)
            self.headline_df = combined_df.drop_duplicates(subset=['url'], keep='last').reset_index(drop=True)

        print(f"📰 Added headlines - total count: {len(self.headline_df)}")

    def get_ai_headlines_count(self) -> int:
        """
        Get count of AI-related headlines in the DataFrame.

        Returns:
            Number of headlines marked as AI-related
        """
        if self.headline_df.empty or 'ai_related' not in self.headline_df.columns:
            return 0

        return int(self.headline_df['ai_related'].sum())

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current session state.

        Returns:
            Dictionary with key metrics and status information
        """
        ai_count = self.get_ai_headlines_count()
        total_headlines = len(self.headline_df)

        return {
            "headlines": {
                "total": total_headlines,
                "ai_related": ai_count,
                "non_ai": total_headlines - ai_count,
                "ai_percentage": f"{(ai_count/total_headlines*100):.1f}%" if total_headlines > 0 else "0%"
            },
            "sources": {
                "config_file": self.sources_file,
                "loaded_sources": len(self.sources),
                "sources_loaded": bool(self.sources)
            },
            "topics": {
                "cluster_topics": len(self.cluster_topics),
                "topics": self.cluster_topics
            },
            "workflow": {
                "max_edits": self.max_edits,
                "edit_complete": self.edit_complete,
                "n_browsers": self.n_browsers
            }
        }

    def print_status(self) -> None:
        """Print a formatted summary of the current session state."""
        status = self.get_status_summary()

        print("\n" + "="*50)
        print("📊 AGENT SESSION STATE SUMMARY")
        print("="*50)

        print(f"📰 Headlines: {status['headlines']['total']} total")
        print(f"   🤖 AI-related: {status['headlines']['ai_related']} ({status['headlines']['ai_percentage']})")
        print(f"   📄 Non-AI: {status['headlines']['non_ai']}")

        print(f"\n📡 Sources: {status['sources']['loaded_sources']} loaded")
        print(f"   📁 Config: {status['sources']['config_file']}")

        print(f"\n🏷️  Topics: {status['topics']['cluster_topics']} cluster topics")
        if status['topics']['topics']:
            print(f"   📋 Topics: {', '.join(status['topics']['topics'])}")

        print(f"\n⚙️  Workflow:")
        print(f"   ✏️  Max edits: {status['workflow']['max_edits']}")
        print(f"   ✅ Edit complete: {status['workflow']['edit_complete']}")
        print(f"   🌐 Browsers: {status['workflow']['n_browsers']}")

        print("="*50 + "\n")


# Example usage and factory functions
def create_default_session() -> AgentSessionState:
    """
    Create a default session state with sensible defaults.

    Returns:
        AgentSessionState configured with default values
    """
    return AgentSessionState()


def create_session_from_config(config_file: str = "session_config.yaml") -> AgentSessionState:
    """
    Create a session state from a configuration file.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        AgentSessionState configured from file

    Expected config file format:
    ```yaml
    sources_file: "sources.yaml"
    max_edits: 5
    n_browsers: 4
    cluster_topics:
      - "AI/ML"
      - "Robotics"
      - "NLP"
    ```
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}

        return AgentSessionState(
            sources_file=config.get('sources_file', 'sources.yaml'),
            cluster_topics=config.get('cluster_topics', []),
            max_edits=config.get('max_edits', 3),
            n_browsers=config.get('n_browsers', 3)
        )

    except FileNotFoundError:
        print(f"⚠️  Config file {config_file} not found, using defaults")
        return create_default_session()


if __name__ == "__main__":
    # Demo usage
    print("🚀 Creating demo session state...")

    # Create session with custom values
    session = AgentSessionState(
        sources_file="sources.yaml",
        cluster_topics=["AI/ML", "Robotics", "NLP", "Computer Vision", "Other"],
        max_edits=5,
        n_browsers=4
    )

    # Add some demo headlines
    demo_headlines = [
        {
            "headline": "AI Is Replacing Online Moderators, But It's Bad at the Job",
            "url": "https://example.com/ai-moderators",
            "source": "tech_news",
            "timestamp": "2024-01-15",
            "ai_related": True
        },
        {
            "headline": "Local Restaurant Opens New Location",
            "url": "https://example.com/restaurant",
            "source": "local_news",
            "timestamp": "2024-01-15",
            "ai_related": False
        }
    ]

    session.add_headlines_to_df(demo_headlines)
    session.print_status()


