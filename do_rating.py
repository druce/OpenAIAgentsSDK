import choix
from typing import List, Tuple, Optional
import numpy as np
import math
import pandas as pd
import sqlite3
from datetime import datetime, timezone, timedelta
from config import NEWSAGENTDB
# from db import Article
import logging
import random
import asyncio
from pydantic import BaseModel, Field

from llm import LLMagent, LangfuseClient, paginate_df_async

_logger = logging.getLogger(__name__)


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


async def run_battles(batch, logger=_logger):
    """
    Run a batch of battles using the LLM model.
    Gets results ranking articles, which we can send to Bradley-Terry

    Args:
        batch: DataFrame containing articles to be battled.

    Returns:
        List of (id1, id2) pairs that were played this round.
    """
    system, user, model = LangfuseClient().get_prompt("newsagent/battle_prompt")

    battle_agent = LLMagent(
        system_prompt=system,
        user_prompt=user,
        output_type=QualityAssessment,
        model=model,
        verbose=False,
        logger=logger
    )

    itemlist = await battle_agent.filter_batch(
        batch,
        value_field='low_quality',
        item_list_field='results_list',
        item_id_field='id',
        chunk_size=25,
        return_probabilities=True
    )

    return [item.id for item in itemlist.items]


async def bradley_terry(headline_df: pd.DataFrame, logger=_logger) -> pd.DataFrame:
    """
    Runs Bradley-Terry rating using the `choix` library.

    Args:
        headline_df (pd.DataFrame): DataFrame containing articles to be ranked.

    Returns:
        pd.DataFrame: DataFrame with additional 'bradley_terry' column containing the computed ratings.
    """

    logger.info("running Bradley-Terry rating")

    # arbitrary n_rounds, around 10 rounds for 100
    # could just continue until ranking_change_sum < number of articles, avg change < 1
    # or until first increase, from e.g. 2 rounds ago, indicating we are converging
    n_rounds = max(2, math.ceil(math.log(len(headline_df))*3-2))
    default_batch_size = 5
    min_batch_size = 2
    target_batches = 10
    jitter_percent = 3.5

    # must ensure canonical sort because prompt will return ids in order
    headline_df = headline_df.sort_values("id")
    headline_df = headline_df.reset_index(drop=True)
    headline_df['id'] = headline_df.index

    # Initialize Bradley-Terry ratings and rankings
    headline_df['bradley_terry'] = 0.0
    previous_rankings = headline_df['bradley_terry'].rank(
        method='min', ascending=False).astype(int)

    batch_size = max(min_batch_size, min(
        default_batch_size, len(headline_df) // target_batches))

    all_battles = []
    all_results = []
    for round_num in range(1, n_rounds + 1):
        logger.info(f"\n--- Running round {round_num}/{n_rounds} ---")

        # sort by current rating + jitter for balanced matchups but also some randomness
        battle_df = headline_df.copy()

        # don't want to battle eg same top 5 each time, add randomness to sort rating, sort order, batch size
        # if you have batch size of 5 and 101 articles, bottom one possibly always skipped
        # or 102 articles, bottom two possibly always just battle each other.
        # Jitter helps but is it enough? don't want to jitter so much that it's not a true reflection of relative quality
        # randomize ascending/descending order and randomize bumping batch size by 1. maybe overkill but regularizes.
        jitter_multipliers = [
            1 + random.uniform(-jitter_percent/100, jitter_percent/100) for _ in range(len(battle_df))]
        battle_df['jittered_rating'] = battle_df['bradley_terry'] * \
            jitter_multipliers
        # to add more jitter randomize sort order by rating, set ascending to random 0 or 1
        battle_df = battle_df.sort_values(
            'jittered_rating', ascending=np.random.randint(0, 2)).drop('jittered_rating', axis=1)

        # Paginate into batches, randomize batch size by 1

        batches = paginate_df_async(
            battle_df[["id", "input_str"]], maxpagelen=max(min_batch_size, batch_size + np.random.randint(0, 3)-1))

        # run_battles for the round (async in parallel)
        tasks = [run_battles(batch) for batch in batches]
        # append results to all_battles
        for battle_result in await asyncio.gather(*tasks):
            for i in range(0, len(battle_result)-1):
                for j in range(i+1, len(battle_result)):
                    all_battles.append((battle_result[i], battle_result[j]))

        # run bradley_terry on results so far
        headline_df['bradley_terry'] = update_ratings_with_choix(
            all_battles, len(headline_df))
        headline_df["bt_z"] = (headline_df["bradley_terry"] - headline_df["bradley_terry"].mean()) / \
            headline_df["bradley_terry"].std(ddof=0)

        new_rankings = headline_df['bradley_terry'].rank(
            method='min', ascending=False).astype(int)
        # sum absolute changes in rankings
        logger.info(f"After round {round_num}/{n_rounds}: ")
        ranking_changes = (previous_rankings != new_rankings).sum()
        logger.info(f"Number of ranking changes: {ranking_changes}")
        ranking_change_sum = np.abs(previous_rankings - new_rankings).sum()
        avg_change = ranking_change_sum / len(headline_df)
        logger.info(
            f"Sum of absolute ranking changes: {ranking_change_sum:.1f} (avg rank chg {avg_change:.2f})")
        all_results.append(avg_change)
        if len(all_results) > 4 and (all_results[-1] + all_results[-2]) > (all_results[-3] + all_results[-4]):
            logger.info("Increase in avg rank change, converging")
            break
        previous_rankings = new_rankings

    return headline_df


class QualityAssessment(BaseModel):
    """Assessment of article quality"""
    low_quality: bool = Field(
        description="Whether the article is low quality (spam, clickbait, etc.)")


class TopicRelevance(BaseModel):
    """Assessment of article topic relevance"""
    on_topic: bool = Field(
        description="Whether the article is on topic for AI/tech news")


class ImportanceAssessment(BaseModel):
    """Assessment of article importance"""
    important: bool = Field(
        description="Whether the article covers important developments")


async def fn_rate_articles(headline_df: pd.DataFrame, model_medium, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
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
        'article_len': 1,
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

    rating_df['input_str'] = rating_df['title'] + "\n" + rating_df['summary']

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
            output_type=QualityAssessment,
            model=model,
            verbose=False,
            logger=logger
        )

        rating_df['low_quality'] = await quality_agent.filter_dataframe(
            rating_df[['id', 'input_str']],
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
        system, user, model = LangfuseClient().get_prompt("newsagent/rate_relevance")

        topic_agent = LLMagent(
            system_prompt=system,
            user_prompt=user,
            output_type=TopicRelevance,
            model=model,
            verbose=False,
            logger=logger
        )

        rating_df['relevant'] = await topic_agent.filter_dataframe(
            rating_df[['id', 'input_str']],
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
        rating_df['relevant'] = 1

    counts = rating_df["relevant"].value_counts().to_dict()
    if logger:
        logger.info(f"relevant articles: {counts}")

    # Importance assessment
    if logger:
        logger.info("Rating importance probability")

    try:
        system, user, model = LangfuseClient().get_prompt("newsagent/rate_importance")

        importance_agent = LLMagent(
            system_prompt=system,
            user_prompt=user,
            output_type=ImportanceAssessment,
            model=model,
            verbose=False,
            logger=logger
        )

        rating_df['important'] = await importance_agent.filter_dataframe(
            rating_df[['id', 'input_str']],
            value_field='important',
            item_list_field='results_list',
            item_id_field='id',
            chunk_size=25
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
    rating_df['adjusted_len'] = np.log10(rating_df['article_len']) - 3
    rating_df['adjusted_len'] = rating_df['adjusted_len'].clip(
        lower=0, upper=2)

    rating_df['rating'] = rating_df['reputation'] \
        + rating_df['adjusted_len'] \
        + rating_df['on_topic'] \
        + rating_df['importance'] \
        - rating_df['low_quality'] \
        + rating_df['bt_z'] \
        + rating_df['recency_score']
    # Filter out low rated articles
    # Note: MINIMUM_STORY_RATING should be defined in config or passed as parameter
    MINIMUM_STORY_RATING = -1.0  # Default threshold

    low_rated_count = len(
        rating_df[rating_df['rating'] < MINIMUM_STORY_RATING])
    if logger:
        logger.info(f"Low rated articles: {low_rated_count}")
        for row in rating_df[rating_df['rating'] < MINIMUM_STORY_RATING].itertuples():
            logger.info(f"low rated article: {row.title} {row.rating}")

    rating_df = rating_df[rating_df['rating'] >= MINIMUM_STORY_RATING].copy()

    # sort by rating
    rating_df = rating_df.sort_values('rating', ascending=False)
    rating_df = rating_df.reset_index(drop=True)
    rating_df['id'] = rating_df.index

    if logger:
        logger.info(f"articles after rating: {len(rating_df)}")

    # Redo bullets with topics and rating (placeholder function)
    # Note: make_bullet function should be imported or defined
    try:
        from utilities import make_bullet
        rating_df["bullet"] = rating_df.apply(make_bullet, axis=1)
    except ImportError:
        if logger:
            logger.warning(
                "make_bullet function not available, setting empty bullets")
        rating_df["bullet"] = ""

    # insert into db to keep a record and eventually train models on summaries
    # Only keep the columns you want to insert
    cols = ['url', 'src', 'site_name', 'hostname',
            'title', 'final_url', 'bullet', 'rating']
    records = rating_df[cols].to_records(index=False)
    rows = list(records)

    conn = sqlite3.connect(NEWSAGENTDB)
    cursor = conn.cursor()
    insert_sql = """
    INSERT INTO daily_summaries
    (url, src, site_name, hostname, title, actual_url, bullet, rating)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(insert_sql, rows)
    conn.commit()
    conn.close()

    return rating_df
