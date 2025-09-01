#!/usr/bin/env python3
"""
Quick Debug - Simple commands for testing agent state and tools
"""

import asyncio
import json
from direct_state_access import DirectStateAccess

def show_article_sample(session_id: str, limit: int = 3):
    """Show sample articles from a session"""
    accessor = DirectStateAccess(session_id)
    state = accessor.load_state()
    
    print(f"📰 Sample Articles from {session_id}")
    print("=" * 50)
    print(f"Current step: {state.current_step}")
    print(f"Total articles: {len(state.headline_data)}")
    
    for i, article in enumerate(state.headline_data[:limit]):
        print(f"\n🔹 Article {i+1}:")
        print(f"   Title: {article.get('title', 'N/A')[:80]}...")
        print(f"   Source: {article.get('source', 'N/A')}")
        print(f"   AI-related: {article.get('ai_related', 'Not set')}")
        if article.get('content'):
            content_preview = article['content'][:100] + "..." if len(article['content']) > 100 else article['content']
            print(f"   Content preview: {content_preview}")
        if article.get('quality_rating'):
            print(f"   Rating: {article.get('quality_rating')}/10")
        if article.get('cluster_topic'):
            print(f"   Cluster: {article.get('cluster_topic')}")

def show_summaries_sample(session_id: str, limit: int = 2):
    """Show sample summaries from a session"""
    accessor = DirectStateAccess(session_id)
    state = accessor.load_state()
    
    print(f"📝 Sample Summaries from {session_id}")
    print("=" * 50)
    
    count = 0
    for url, summary in state.article_summaries.items():
        if count >= limit:
            break
        
        # Find the article title
        article_title = "Unknown"
        for article in state.headline_data:
            if article.get('url') == url:
                article_title = article.get('title', 'Unknown')[:60]
                break
        
        print(f"\n🔹 Summary {count+1}:")
        print(f"   Article: {article_title}...")
        print(f"   Summary:")
        for bullet in summary:
            print(f"     • {bullet}")
        
        count += 1

def show_clusters(session_id: str):
    """Show topic clusters from a session"""
    accessor = DirectStateAccess(session_id)
    state = accessor.load_state()
    
    print(f"🏷️  Topic Clusters from {session_id}")
    print("=" * 50)
    
    for topic, urls in state.topic_clusters.items():
        print(f"\n📂 {topic}: {len(urls)} articles")
        # Show first few article titles
        count = 0
        for url in urls[:3]:
            for article in state.headline_data:
                if article.get('url') == url:
                    title = article.get('title', 'Unknown')[:50]
                    print(f"   • {title}...")
                    count += 1
                    break
        if len(urls) > 3:
            print(f"   ... and {len(urls) - 3} more")

def show_newsletter_preview(session_id: str):
    """Show newsletter preview from a session"""
    accessor = DirectStateAccess(session_id)
    state = accessor.load_state()
    
    print(f"📰 Newsletter Preview from {session_id}")
    print("=" * 50)
    
    if state.final_newsletter:
        preview_length = 500
        preview = state.final_newsletter[:preview_length]
        if len(state.final_newsletter) > preview_length:
            preview += "..."
        
        print(preview)
        print(f"\n📊 Total length: {len(state.final_newsletter)} characters")
        print(f"📊 Word count: {len(state.final_newsletter.split())} words")
    else:
        print("No final newsletter found.")

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Quick Debug Commands:")
        print("  python quick_debug.py <session_id> articles [limit]    - Show sample articles")
        print("  python quick_debug.py <session_id> summaries [limit]   - Show sample summaries") 
        print("  python quick_debug.py <session_id> clusters            - Show topic clusters")
        print("  python quick_debug.py <session_id> newsletter          - Show newsletter preview")
        print("  python quick_debug.py <session_id> all                 - Show everything")
        print("")
        print("Available sessions: debug_step_by_step, simple_test, etc.")
        return
    
    session_id = sys.argv[1]
    command = sys.argv[2]
    
    if command == "articles":
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        show_article_sample(session_id, limit)
        
    elif command == "summaries":
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        show_summaries_sample(session_id, limit)
        
    elif command == "clusters":
        show_clusters(session_id)
        
    elif command == "newsletter":
        show_newsletter_preview(session_id)
        
    elif command == "all":
        show_article_sample(session_id, 2)
        print("\n")
        show_summaries_sample(session_id, 2)
        print("\n")
        show_clusters(session_id)
        print("\n")
        show_newsletter_preview(session_id)
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()