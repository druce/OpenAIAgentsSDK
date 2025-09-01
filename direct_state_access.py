#!/usr/bin/env python3
"""
Direct State Access - Bypass agent and directly inspect/modify state
"""

import json
import sqlite3
import pickle
from typing import Any, Dict
from mock_agent import NewsletterAgentState

class DirectStateAccess:
    """Direct access to SQLite session state without going through the agent"""
    
    def __init__(self, session_id: str, db_path: str = "newsletter_agent.db"):
        self.session_id = session_id
        self.db_path = db_path
    
    def load_state(self) -> NewsletterAgentState:
        """Load state directly from SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query the session data
            cursor.execute("SELECT data FROM sessions WHERE id = ?", (self.session_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                # Deserialize the state
                state_data = pickle.loads(result[0])
                return NewsletterAgentState(**state_data)
            else:
                print(f"No state found for session: {self.session_id}")
                return NewsletterAgentState()
        
        except Exception as e:
            print(f"Error loading state: {e}")
            return NewsletterAgentState()
    
    def save_state(self, state: NewsletterAgentState):
        """Save state directly to SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    data BLOB
                )
            """)
            
            # Serialize and save state
            state_data = pickle.dumps(state.model_dump())
            cursor.execute(
                "INSERT OR REPLACE INTO sessions (id, data) VALUES (?, ?)",
                (self.session_id, state_data)
            )
            
            conn.commit()
            conn.close()
            print(f"State saved for session: {self.session_id}")
            
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def inspect_headline_data(self, limit: int = 5) -> None:
        """Inspect headline_data in detail"""
        state = self.load_state()
        
        print(f"📰 HEADLINE DATA INSPECTION")
        print(f"Total articles: {len(state.headline_data)}")
        print(f"Current step: {state.current_step}")
        
        if state.headline_data:
            print(f"\nFirst {limit} articles:")
            for i, article in enumerate(state.headline_data[:limit]):
                print(f"\nArticle {i+1}:")
                print(f"  Title: {article.get('title', 'N/A')[:100]}...")
                print(f"  Source: {article.get('source', 'N/A')}")
                print(f"  URL: {article.get('url', 'N/A')[:60]}...")
                print(f"  AI-related: {article.get('ai_related', 'Not set')}")
                print(f"  Has content: {bool(article.get('content'))}")
                print(f"  Quality rating: {article.get('quality_rating', 'Not set')}")
                print(f"  Cluster: {article.get('cluster_topic', 'Not set')}")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state"""
        state = self.load_state()
        
        return {
            "current_step": state.current_step,
            "workflow_complete": state.workflow_complete,
            "total_articles": len(state.headline_data),
            "ai_related_articles": sum(1 for a in state.headline_data if a.get('ai_related') is True),
            "articles_with_content": sum(1 for a in state.headline_data if a.get('content')),
            "articles_with_ratings": sum(1 for a in state.headline_data if a.get('quality_rating')),
            "unique_sources": len(set(a.get('source', 'Unknown') for a in state.headline_data)),
            "summaries_count": len(state.article_summaries),
            "clusters_count": len(state.topic_clusters),
            "sections_count": len(state.newsletter_sections),
            "has_final_newsletter": bool(state.final_newsletter)
        }
    
    def reset_to_step(self, target_step: int):
        """Reset state to a specific step (useful for testing)"""
        state = self.load_state()
        
        # Reset step
        state.current_step = target_step
        state.workflow_complete = False
        
        # Clear data for steps after target_step
        if target_step < 4:
            state.article_summaries = {}
        if target_step < 5:
            state.topic_clusters = {}
        if target_step < 7:
            state.newsletter_sections = {}
        if target_step < 9:
            state.final_newsletter = ""
            
        # Reset article properties for steps after target_step
        for article in state.headline_data:
            if target_step < 2:
                article.pop('ai_related', None)
            if target_step < 3:
                article.pop('content', None)
                article.pop('download_timestamp', None)
                article.pop('content_length', None)
            if target_step < 4:
                article.pop('summary_bullets', None)
                article.pop('summary_timestamp', None)
            if target_step < 5:
                article.pop('cluster_topic', None)
                article.pop('cluster_timestamp', None)
            if target_step < 6:
                article.pop('quality_rating', None)
                article.pop('rating_timestamp', None)
        
        self.save_state(state)
        print(f"Reset state to step {target_step}")

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python direct_state_access.py <session_id> <command> [args]")
        print("")
        print("Commands:")
        print("  summary                    - Show state summary")
        print("  inspect [limit]           - Inspect headline data")
        print("  reset_to <step>           - Reset to specific step")
        print("  export <filename>         - Export state to JSON")
        print("")
        print("Examples:")
        print("  python direct_state_access.py simple_test summary")
        print("  python direct_state_access.py simple_test inspect 10")
        print("  python direct_state_access.py simple_test reset_to 1")
        return
    
    session_id = sys.argv[1]
    command = sys.argv[2]
    
    accessor = DirectStateAccess(session_id)
    
    if command == "summary":
        summary = accessor.get_state_summary()
        print("📊 STATE SUMMARY")
        print("=" * 30)
        for key, value in summary.items():
            print(f"{key}: {value}")
    
    elif command == "inspect":
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        accessor.inspect_headline_data(limit)
    
    elif command == "reset_to":
        target_step = int(sys.argv[3])
        accessor.reset_to_step(target_step)
    
    elif command == "export":
        filename = sys.argv[3]
        state = accessor.load_state()
        with open(filename, 'w') as f:
            json.dump(state.model_dump(), f, indent=2, default=str)
        print(f"State exported to {filename}")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()