import choix
from typing import List, Tuple, Optional
import numpy as np
import math
import pandas as pd
import sqlite3
from datetime import datetime, timezone, timedelta
from config import NEWSAGENTDB
from db import Article
import logging
# import random
import asyncio
from pydantic import BaseModel, Field

from llm import LLMagent, LangfuseClient  # , paginate_df_async

_logger = logging.getLogger(__name__)


# class QualityAssessment(BaseModel):
#     """Assessment of article quality"""
#     low_quality: bool = Field(
#         description="Whether the article is low quality (spam, clickbait, etc.)")


# class TopicRelevance(BaseModel):
#     """Assessment of article topic relevance"""
#     on_topic: bool = Field(
#         description="Whether the article is on topic for AI/tech news")


# class ImportanceAssessment(BaseModel):
#     """Assessment of article importance"""
#     important: bool = Field(
#         description="Whether the article covers important developments")


class StoryRating(BaseModel):
    """StoryRating class for generic structured output rating"""
    id: int = Field(description="The id of the story")
    rating: int = Field(description="An integer rating of the story")


class StoryRatings(BaseModel):
    """StoryRatings class for structured output filtering of a list of Story"""
    items: List[StoryRating] = Field(description="List of StoryRating")


class StoryOrder(BaseModel):
    """StoryOrder class for generic structured output rating"""
    id: int = Field(description="The id of the story")


class StoryOrderList(BaseModel):
    """List of StoryOrder for structured output"""
    items: List[StoryOrder] = Field(
        description="List of StoryOrder")


def update_ratings_with_choix(all_battles: List[Tuple[int, int]], num_items: int, logger=_logger) -> np.ndarray:
    """
    Use choix to compute Bradley-Terry ratings based on a series of pairwise comparisons
    Similar to ELO but more mathematically optimal based on full hostory
    Unlike ELO, can't update online after 1 match

    Args:
        all_battles: List of (winner_id, loser_id) tuples
        num_items: Total number of items being ranked

    Returns:
        Array of Bradley-Terry ratings computed by choix
    """
    choix_battles = [(int(winner), int(loser))
                     for winner, loser in all_battles]

    try:
        # Compute Bradley-Terry ratings using maximum likelihood estimation
        ratings = choix.opt_pairwise(num_items, choix_battles)
        return ratings
    except ValueError as e:
        logger.warning(
            f"Warning: choix optimization failed - invalid data ({e}), returning zeros")
        return np.zeros(num_items)
    except RuntimeError as e:
        logger.warning(
            f"Warning: choix optimization failed - runtime error ({e}), returning zeros")
        return np.zeros(num_items)
    except ImportError as e:
        print(f"Warning: choix library not available ({e}), returning zeros")
        return np.zeros(num_items)


def swiss_pairing(headline_df: pd.DataFrame, battle_history: np.ndarray) -> List[Tuple[int, int]]:
    """
    Create Swiss-style pairings for Bradley-Terry battles.
    Returns all valid pairs based on current ranking.

    Args:
        headline_df: DataFrame sorted by current rating (best first)
        battle_history: NxN matrix where [i,j]=1 means i vs j already battled

    Returns:
        List of (story1_id, story2_id) pairs for this round
    """
    used_this_round = set()
    pairs = []

    # Sort by current rating (highest first)
    sorted_df = headline_df.sort_values(
        'bradley_terry', ascending=False).reset_index(drop=True)

    for i, row in sorted_df.iterrows():
        if row['id'] in used_this_round:
            continue

        story1_id = row['id']

        # Find best available opponent (highest rated, not yet battled)
        # Start from i+1 to avoid self-pairing and ensure unique pairs
        for j in range(i + 1, len(sorted_df)):
            opponent_row = sorted_df.iloc[j]
            opponent_id = opponent_row['id']

            if (opponent_id not in used_this_round and
                    battle_history[story1_id, opponent_id] == 0):

                pairs.append((story1_id, opponent_id))
                used_this_round.add(story1_id)
                used_this_round.add(opponent_id)
                break

    return pairs


async def run_battle_pair(story1: str, story2: str, agent,
                          semaphore: asyncio.Semaphore) -> int:
    """
    Battle two stories and return winner ID.

    Returns:
        0 if story1 wins, 1 if story2 wins
    """
    async with semaphore:
        prompt = f"""Compare these two stories and determine which is higher quality:

Story A: {story1}

Story B: {story2}

Return only "A" if Story A is better, or "B" if Story B is better."""

        result = await agent.run_prompt(prompt)
        return 0 if result.strip().upper() == 'A' else 1


async def process_battle_round(pairs: List[Tuple[int, int]],
                               headline_df: pd.DataFrame,
                               agent,
                               max_concurrent: int = 10,
                               logger=_logger) -> List[Tuple[int, int]]:
    """
    Process all battle pairs concurrently and return winner/loser tuples.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create battle tasks
    tasks = []
    for story1_id, story2_id in pairs:
        story1_text = headline_df.loc[headline_df['id']
                                      == story1_id, 'input_text'].iloc[0]
        story2_text = headline_df.loc[headline_df['id']
                                      == story2_id, 'input_text'].iloc[0]
        task = run_battle_pair(story1_text, story2_text, agent, semaphore)
        tasks.append((task, story1_id, story2_id))

    # Execute all battles concurrently
    results = []
    for task, story1_id, story2_id in tasks:
        winner_idx = await task
        if winner_idx == 0:
            results.append((story1_id, story2_id))  # story1 wins
        else:
            results.append((story2_id, story1_id))  # story2 wins

        logger.info(
            f"Battle: {story1_id} vs {story2_id} -> winner: {story1_id if winner_idx == 0 else story2_id}")

    return results


async def bradley_terry(headline_df: pd.DataFrame, logger=_logger) -> pd.DataFrame:
    """
    Enhanced Bradley-Terry rating with Swiss pairing and async battles.

    Args:
        headline_df (pd.DataFrame): DataFrame containing articles to be ranked.

    Returns:
        pd.DataFrame: DataFrame with additional 'bradley_terry' column containing the computed ratings.
    """
    logger.info("Running Bradley-Terry rating with Swiss pairing")

    # Setup
    n_rounds = max(2, math.ceil(math.log(len(headline_df)) * 3 - 2))
    max_concurrent = 10

    # must ensure canonical sort because prompt will return ids in order
    headline_df = headline_df.sort_values("rating", ascending=False)
    headline_df = headline_df.reset_index(drop=True)
    headline_df['id'] = headline_df.index

    # Initialize Bradley-Terry ratings and rankings
    headline_df['bradley_terry'] = 1-(headline_df["id"]+1)/1000
    previous_rankings = (headline_df["id"]+1).to_list()

    # Battle history matrix (N x N)
    n_stories = len(headline_df)
    battle_history = np.zeros((n_stories, n_stories), dtype=int)

    # Setup battle agent
    system, user, model = LangfuseClient().get_prompt("newsagent/battle_prompt")
    battle_agent = LLMagent(
        system_prompt=system,
        user_prompt=user,
        model=model,
        output_type=StoryOrder,
        verbose=False,
        logger=logger
    )
    all_battles = []
    all_results = []

    for round_num in range(1, n_rounds + 1):
        logger.info(f"\n--- Running round {round_num}/{n_rounds} ---")

        # Swiss pairing
        pairs = swiss_pairing(headline_df, battle_history)
        logger.info(f"Generated {len(pairs)} battle pairs")

        if not pairs:
            logger.info("No more valid pairings available")
            break

        all_ids = []
        for pair in pairs:
            all_ids.extend(list(pair))
        all_ids_set = set(all_ids)

        for row in headline_df.itertuples():
            if row.id not in all_ids_set:
                all_ids.append(row.index)

        # Process battles concurrently
        battle_results = await process_battle_round(pairs, headline_df, battle_agent, max_concurrent, logger)

        # Update battle history and results
        for winner_id, loser_id in battle_results:
            all_battles.append((winner_id, loser_id))
            battle_history[winner_id, loser_id] = 1
            battle_history[loser_id, winner_id] = 1

        # Update ratings
        headline_df['bradley_terry'] = update_ratings_with_choix(
            all_battles, len(headline_df), logger)
        headline_df["bt_z"] = (headline_df["bradley_terry"] - headline_df["bradley_terry"].mean()) / \
            headline_df["bradley_terry"].std(ddof=0)

        # Check convergence
        new_rankings = headline_df['bradley_terry'].rank(
            method='min', ascending=False).astype(int)
        ranking_changes = (previous_rankings != new_rankings).sum()
        ranking_change_sum = np.abs(previous_rankings - new_rankings).sum()
        avg_change = ranking_change_sum / len(headline_df)

        logger.info(f"After round {round_num}/{n_rounds}: ")
        logger.info(f"Number of ranking changes: {ranking_changes}")
        logger.info(
            f"Sum of absolute ranking changes: {ranking_change_sum:.1f} (avg rank chg {avg_change:.2f})")

        all_results.append(avg_change)

        # Convergence detection
        if avg_change < 0.1:  # Convergence threshold
            logger.info("Converged - stopping early")
            break
        elif len(all_results) > 4 and (all_results[-1] + all_results[-2]) > (all_results[-3] + all_results[-4]):
            logger.info("Increase in avg rank change, converging")
            break

        previous_rankings = new_rankings

    return headline_df


async def fn_rate_articles(headline_df: pd.DataFrame, logger: Optional[logging.Logger] = None, minimum_story_rating: float = -1.0) -> pd.DataFrame:
    """
    Calculate ratings for articles using LLM-based assessments and Bradley-Terry ranking.

    Args:
        headline_df: DataFrame containing articles to be rated
        model_medium: Model to use for Bradley-Terry comparisons
        logger: Optional logger for tracking progress

    Returns:
        DataFrame with additional rating columns and filtered by minimum rating threshold

    The rating process includes:
    1. Ask yes/no questions: low quality (spammy), on topic, importance
    2. Use Bradley-Terry to rate articles based on pairwise comparisons
    3. Add points for recency
    4. Add points for log length (more in-depth articles get more points)
    5. Add points for reputation
    """
    rating_df = headline_df.copy().fillna({
        'content_length': 1,
        'reputation': 0,
        'on_topic': 0,
        'importance': 0,
        'low_quality': 0,
    })
    if logger:
        logger.info(
            f"Calculating article ratings for {len(rating_df)} articles")
    # Ensure 'title' and 'summary' are always strings
    rating_df['title'] = rating_df['title'].fillna("")
    rating_df['title'] = rating_df['title'].astype(str)
    rating_df['summary'] = rating_df['summary'].astype(str)
    rating_df['summary'] = rating_df['summary'].fillna("")

    rating_df['input_text'] = rating_df['title'] + "\n" + rating_df['summary']
    if logger:
        logger.info("Rating recency")
    # add points for recency
    yesterday = (datetime.now(timezone.utc)
                 - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rating_df['last_updated'] = rating_df['last_updated'].fillna(yesterday)
    rating_df["age"] = (datetime.now(timezone.utc) -
                        pd.to_datetime(rating_df['last_updated']))
    rating_df["age"] = rating_df["age"].dt.total_seconds() / (24 * 60 * 60)
    rating_df["age"] = rating_df["age"].clip(lower=0)  # no negative dates
    # only consider articles from the last week
    rating_df = rating_df[rating_df["age"] < 7].copy()
    k = np.log(2)  # 1/2 after 1 day
    # 1 point at age 0, 0 at age 1, -0.5 at age 2, -1 at age infinity
    rating_df["recency_score"] = 2 * np.exp(-k * rating_df["age"]) - 1

    # Low quality assessment using LLM
    if logger:
        logger.info("Rating spam probability")

    # Get prompt from Langfuse
    try:
        system, user, model = LangfuseClient().get_prompt("newsagent/rate_quality")

        quality_agent = LLMagent(
            system_prompt=system,
            user_prompt=user,
            output_type=StoryRatings,
            model=model,
            verbose=False,
            logger=logger
        )

        rating_df['low_quality'] = await quality_agent.filter_dataframe(
            rating_df[['id', 'input_text']],
            value_field='low_quality',
            item_list_field='results_list',
            item_id_field='id',
            chunk_size=25,
            return_probabilities=True
        )

    except Exception as e:
        if logger:
            logger.warning(
                f"Failed to get quality assessment prompt from Langfuse: {e}")
        # Fallback: assume no articles are low quality
        rating_df['low_quality'] = 0

    counts = rating_df["low_quality"].value_counts().to_dict()
    if logger:
        logger.info(f"low quality articles: {counts}")

    # Topic relevance assessment
    try:
        system, user, model = LangfuseClient().get_prompt("newsagent/rate_on_topic")

        topic_agent = LLMagent(
            system_prompt=system,
            user_prompt=user,
            output_type=StoryRatings,
            model=model,
            verbose=False,
            logger=logger
        )

        rating_df['on_topic'] = await topic_agent.filter_dataframe(
            rating_df[['id', 'input_text']],
            value_field='on_topic',
            item_list_field='results_list',
            item_id_field='id',
            chunk_size=25,
            return_probabilities=True
        )

    except Exception as e:
        if logger:
            logger.warning(
                f"Failed to get topic relevance prompt from Langfuse: {e}")
        # Fallback: assume all articles are on topic
        rating_df['on_topic'] = 1

    counts = rating_df["on_topic"].value_counts().to_dict()
    if logger:
        logger.info(f"on topic articles: {counts}")

    # Importance assessment
    if logger:
        logger.info("Rating importance probability")

    try:
        system, user, model = LangfuseClient().get_prompt("newsagent/rate_importance")

        importance_agent = LLMagent(
            system_prompt=system,
            user_prompt=user,
            output_type=StoryRatings,
            model=model,
            verbose=False,
            logger=logger
        )

        rating_df['important'] = await importance_agent.filter_dataframe(
            rating_df[['id', 'input_text']],
            value_field='important',
            item_list_field='results_list',
            item_id_field='id',
            chunk_size=25,
            return_probabilities=True
        )

    except Exception as e:
        if logger:
            logger.warning(
                f"Failed to get importance assessment prompt from Langfuse: {e}")
        # Fallback: assume all articles are important
        rating_df['important'] = 1

    counts = rating_df["important"].value_counts().to_dict()
    if logger:
        logger.info(f"important articles: {counts}")

    # AI is good at yes or no questions, not at converting understanding to a numerical rating.
    # Use Bradley-Terry to rate articles based on a series of pairwise comparisons
    # Note: Bradley-Terry rating is commented out for now due to complexity and cost
    # if logger:
    #     logger.info("running Bradley-Terry rating using head-to-head prompt comparisons")
    # aidf = await bradley_terry(aidf, model=model_medium)

    # For now, set bt_z to 0 as placeholder
    rating_df['bt_z'] = 0.0

    # could test if the prompts bias for/against certain types of stories, adjust the prompts, or boost ratings if they match those topics
    # bonus for longer articles
    # len < 1000 -> 0
    # len > 10000 -> 1
    rating_df['adjusted_len'] = rating_df['content_length'].clip(lower=1)
    rating_df['adjusted_len'] = np.log10(rating_df['adjusted_len']) - 3
    rating_df['adjusted_len'] = rating_df['adjusted_len'].clip(
        lower=0, upper=2)

    rating_df['rating'] = rating_df['reputation'] \
        + rating_df['adjusted_len'] \
        + rating_df['on_topic'] \
        + rating_df['important'] \
        - rating_df['low_quality'] \
        + rating_df['bt_z'] \
        + rating_df['recency_score']
    # Filter out low rated articles

    low_rated_count = len(
        rating_df[rating_df['rating'] < minimum_story_rating])
    if logger:
        logger.info(f"Low rated articles: {low_rated_count}")
        for row in rating_df[rating_df['rating'] < minimum_story_rating].itertuples():
            logger.info(f"low rated article: {row.title} {row.rating}")

    rating_df = rating_df[rating_df['rating'] >= minimum_story_rating].copy()

    # sort by rating
    rating_df = rating_df.sort_values('rating', ascending=False)
    rating_df = rating_df.reset_index(drop=True)
    rating_df['id'] = rating_df.index

    if logger:
        logger.info(f"articles after rating: {len(rating_df)}")

    # Insert articles into database using Article schema
    if logger:
        logger.info(f"Inserting {len(rating_df)} articles into database")

    try:
        with sqlite3.connect(NEWSAGENTDB) as conn:
            Article.create_table(conn)  # Ensure articles table exists

            articles_inserted = 0
            for _, row in rating_df.iterrows():
                try:
                    # Create Article instance with available data
                    article = Article(
                        final_url=row.get('final_url', row.get('url', '')),
                        url=row.get('url', ''),
                        source=row.get('source', row.get('src', '')),
                        title=row.get('title', ''),
                        published=pd.to_datetime(row.get('published')) if pd.notna(
                            row.get('published')) else None,
                        rss_summary=row.get('rss_summary', ''),
                        # Assume True since these went through AI filtering
                        isAI=bool(row.get('isAI', True)),
                        status=row.get('status', 'rated'),
                        html_path=row.get('html_path', ''),
                        last_updated=pd.to_datetime(row.get('last_updated')) if pd.notna(
                            row.get('last_updated')) else None,
                        text_path=row.get('text_path', ''),
                        content_length=int(row.get('content_length', 0)),
                        summary=row.get('summary', ''),
                        description=row.get('description', ''),
                        rating=float(row.get('rating', 0.0)),
                        cluster_label=row.get(
                            'cluster_label', row.get('cluster_name', '')),
                        domain=row.get('domain', row.get('hostname', '')),
                        site_name=row.get('site_name', ''),
                        reputation=float(row.get('reputation', 0.0)) if pd.notna(
                            row.get('reputation')) else None,
                        date=pd.to_datetime(row.get('date')) if pd.notna(
                            row.get('date')) else None
                    )

                    # Use upsert to avoid conflicts on duplicate final_url
                    article.upsert(conn)
                    articles_inserted += 1

                except Exception as e:
                    if logger:
                        logger.warning(
                            f"Failed to insert article {row.get('title', 'Unknown')}: {e}")

            if logger:
                logger.info(
                    f"Successfully inserted {articles_inserted}/{len(rating_df)} articles")

    except Exception as e:
        if logger:
            logger.error(f"Database insertion failed: {e}")

    return rating_df
