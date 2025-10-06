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
from IPython.display import display

from llm import LLMagent, LangfuseClient  # , paginate_df_async

BT_BATCH = 6

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

        story1_id = int(row['id'])

        # Find best available opponent (highest rated, not yet battled)
        # Start from i+1 to avoid self-pairing and ensure unique pairs
        for j in range(i + 1, len(sorted_df)):
            opponent_row = sorted_df.iloc[j]
            opponent_id = int(opponent_row['id'])

            if (opponent_id not in used_this_round and
                    battle_history[story1_id, opponent_id] == 0):

                pairs.append((story1_id, opponent_id))
                used_this_round.add(story1_id)
                used_this_round.add(opponent_id)
                break

    return pairs


async def bt_paginate_list_async(lst, chunk_size: int = 25):
    """Async generator for list pagination with smart last chunk handling."""
    i = 0
    total_len = len(lst)

    while i < total_len:
        remaining = total_len - i

        # If remaining rows are <= chunk_size * 2, split into 2 chunks
        if remaining <= chunk_size * 2 and remaining > chunk_size:
            # Split remaining rows into 2 roughly equal chunks
            first_chunk_size = remaining // 2
            second_chunk_size = remaining - first_chunk_size

            # Yield first chunk
            yield lst[i:i + first_chunk_size]
            await asyncio.sleep(0)

            # Yield second chunk
            yield lst[i + first_chunk_size:i + first_chunk_size + second_chunk_size]
            await asyncio.sleep(0)

            break  # We're done
        else:
            # Normal chunk or final small chunk
            yield lst[i:i + chunk_size]
            await asyncio.sleep(0)
            i += chunk_size


async def process_battle_round(bt_df,
                               battle_order,
                               battle_agent,
                               max_concurrent=100,
                               logger=_logger) -> List[Tuple[int, int]]:
    """
    Process all battle pairs concurrently and return winner/loser tuples.
    """

    async def process_single_batch(batch):
        """Process a single batch with semaphore control."""
        async with semaphore:
            try:
                result = await battle_agent.run_prompt(input_text=str(batch))
                return result
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                return None

    semaphore = asyncio.Semaphore(max_concurrent)

    # Create a DataFrame from the list with the desired order and sort by that
    logger.info("Creating battle order")
    logger.info(f"All ids: {battle_order}")
    order_df = pd.DataFrame(
        {'id': battle_order, 'order': range(len(battle_order))})
    # display(order_df)
    # print(bt_df.columns)

    merge_df = order_df.merge(bt_df, on='id')
    # print(merge_df.columns)
    merge_df = merge_df.sort_values('order')
    # display(merge_df[["id", "order", "input_text"]])

    # Create battle tasks
    records = merge_df[["id", "input_text"]].to_dict('records')
    tasks = []
    BT_BATCH = 6
    async for batch in bt_paginate_list_async(records, BT_BATCH):
        tasks.append(process_single_batch(batch))

    # Execute all battles concurrently
    logger.info(
        f"Processing {len(tasks)} battles of size {BT_BATCH} with concurrency = {max_concurrent}")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Battles complete")
    retlist = []
    for lst in results:
        battles = []
        for item in lst.items:
            battles.append(item.id)
        n_results = len(battles)
        for i in range(n_results-1):
            for j in range(i+1, n_results):
                winner = battles[i]
                loser = battles[j]
                # print(f"Battle: {winner} beat {loser}")
                retlist.append((winner, loser))

    return retlist


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
    # Initialize Bradley-Terry ratings and rankings based on index order
    bt_df = headline_df.copy()
    previous_rankings = np.array(list(range(1, len(bt_df)+1)))
    # dummy ratings based on index order
    bt_df['bradley_terry'] = 1 - previous_rankings / len(bt_df)

    # max theoretical number of rounds to have everyone play everyone
    # if we have n=120 stories, n-1 pairs to play everyone
    # we ask prompt to rank batches
    # each round each plays BT_BATCH-1 battles
    # so we need (n-1) / (BT_BATCH-1) rounds
    # but note that in 1 batch of 6, each plays 5, 2 are guaranteed new, 3 are close in rank
    n_stories = len(bt_df)
    max_rounds = math.ceil((n_stories-1) / (BT_BATCH-1))
    logger.info(
        f"Max {max_rounds} rounds")
    # Battle history matrix (N x N)
    # matrix of battles, did 2 stories battle each other?
    battle_history = np.zeros((n_stories, n_stories), dtype=int)

    # Setup battle agent
    system, user, model = LangfuseClient().get_prompt("newsagent/battle_prompt")
    battle_agent = LLMagent(
        system_prompt=system,
        user_prompt=user,
        model=model,
        output_type=StoryOrderList,
        verbose=False,
        logger=logger
    )
    all_battles = []
    all_results = []

    convergence_threshold = n_stories // 100
    min_rounds = max_rounds // 4

    # run rounds to ensure everyone battles at least once
    for round_num in range(1, max_rounds+1):
        logger.info(
            "---------------------------------------------------")
        logger.info(
            f"Running round {round_num} of {max_rounds}")
        logger.info(
            "---------------------------------------------------")

        # Swiss pairing
        pairs = swiss_pairing(bt_df, battle_history)
        logger.info(f"Generated {len(pairs)} battle pairs")

        if not pairs:
            logger.info("No more valid pairings available")
            break

        # swiss pairing finds pairs that haven't battled before.
        # then we output all pairs in order , randomize any remaining
        # (odd one or ones that couldn't be paired with anyone they haven't battled).
        # then we batch into groups of 6. so we are guaranteed 3 pairs per batch that haven't battled.

        # a better algorithm would be to find the maximum number of never-battled pairs that are close in rank
        # using maximum weight bipartite matching

        # def optimal_swiss_pairing(headline_df, battle_history):
        #     # Create graph of available pairings
        #     G = nx.Graph()
        #     for i in range(len(headline_df)):
        #         for j in range(i+1, len(headline_df)):
        #             # only link those that haven't battled
        #             # alternatively could link all and divide by 2^number of battles
        #             # but need to make it so e.g. 50-rank difference is same as 1 extra battle
        #             if battle_history[i, j] == 0:  # Haven't battled
        #                 # Weight by rating similarity for better matches
        #                 weight = 1.0 / (abs(headline_df.iloc[i]['bradley_terry'] - headline_df.iloc[j]['bradley_terry'])) + 0.1
        #                 G.add_edge(i, j, weight=weight)

        #     # Find maximum weight matching
        #     matching = nx.max_weight_matching(G)
        #     return list(matching)

        all_ids = []
        for pair in pairs:
            all_ids.extend(list(pair))
        all_ids_set = set(all_ids)

        # may have already battled (or odd one out), but add anyway, to ensure everyone battles at every round
        duped_ids = [row.id for row in bt_df.itertuples()
                     if row.id not in all_ids_set]
        np.random.shuffle(duped_ids)
        all_ids.extend(duped_ids)

        logger.info(
            f"len(all_ids): {len(set(all_ids))} ; len(bt_df): {len(bt_df)}")

        # dupe_count = bt_df["id"].duplicated().sum()
        # logger.warning(f"Found {dupe_count} duplicate articles")

        # Process battles concurrently
        battle_results = await process_battle_round(bt_df,
                                                    all_ids,
                                                    battle_agent,
                                                    max_concurrent=1000,
                                                    logger=logger)

        # dupe_count = bt_df["id"].duplicated().sum()
        # logger.warning(f"Found {dupe_count} duplicate articles")

        # Update battle history and results
        all_battles.extend(battle_results)
        logger.info(f"total battles: {len(all_battles)}")
        for winner_id, loser_id in battle_results:
            battle_history[winner_id, loser_id] += 1
            battle_history[loser_id, winner_id] += 1

        # Use choix to compute Bradley-Terry ratings based on a series of pairwise comparisons
        # Similar to ELO but more mathematically optimal based on full hostory
        # Unlike ELO, can't update online after 1 match
        # IMPORTANT: uses sort order , requres index==id
        logger.info("Recomputing Bradley-Terry ratings")
        bt_df['bradley_terry'] = choix.opt_pairwise(len(bt_df), all_battles)
        logger.info("Recomputed Bradley-Terry ratings")

        bt_df["bt_z"] = (bt_df["bradley_terry"] - bt_df["bradley_terry"].mean()) / \
            bt_df["bradley_terry"].std(ddof=0)
        logger.info("Computed Bradley-Terry z-scores")

        # Check convergence
        # Show top 10 ids after sorting by Bradley-Terry rating
        sorted_df = bt_df.copy().sort_values(
            'bradley_terry', ascending=False)
        top_10_ids = sorted_df['id'].tolist()[:10]
        logger.info(
            f"Top 10 ids: {top_10_ids}")

        new_rankings = sorted_df["id"].values
        ranking_changes = (previous_rankings != new_rankings).sum()
        ranking_change_sum = np.abs(previous_rankings - new_rankings).sum()
        avg_change = ranking_change_sum / len(bt_df)
        previous_rankings = new_rankings

        logger.info(f"After round {round_num}/{max_rounds}: ")
        logger.info(f"Number of ranking changes: {ranking_changes}")
        logger.info(
            f"Sum of absolute ranking changes: {ranking_change_sum:.1f} (avg rank chg {avg_change:.2f})")

        all_results.append(avg_change)

        # Convergence detection
        if len(all_results) > min_rounds:  # do at least 1/4 of max rounds
            last_two = all_results[-1] + all_results[-2]
            prev_two = all_results[-3] + all_results[-4]
            if (last_two) < convergence_threshold * 2:
                logger.info("Convergence threshold achieved - stopping")
                break
            else:
                if last_two > prev_two:
                    logger.info("Increase in avg rank change, stopping")
                    break

    return bt_df


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

    # fix bad strings that don't start with 20
    rating_df['last_updated'] = rating_df['last_updated'].apply(
        lambda s: s if isinstance(s, str) and s[:2] == '20' else None)
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
