import choix
from typing import List, Tuple, Optional
import numpy as np
import math
import pandas as pd
from datetime import datetime, timezone, timedelta
import logging
# import random
import asyncio
from pydantic import BaseModel, Field
from IPython.display import display
# , paginate_df_async
from llm import LLMagent, get_langfuse_client, run_prompt_on_dataframe

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


async def swiss_batching(headline_df: pd.DataFrame, battle_history: np.ndarray,
                         batch_size: int = BT_BATCH, min_batch_size: int = 3) -> List[List[int]]:
    """
    Create Swiss-style batches for Bradley-Terry battles.

    Constructs batches where items are close in rank and haven't battled each other.
    This is superior to pairwise matching because it guarantees no repeat battles
    within each batch and maximizes information gain per LLM call.

    Args:
        headline_df: DataFrame with 'id' and 'bradley_terry' columns
        battle_history: NxN matrix where [i,j]!=0 means i vs j already battled
        batch_size: Target batch size (default: 6)
        min_batch_size: Minimum viable batch size (default: 3)

    Returns:
        List of batches, where each batch is a list of story IDs

    Algorithm:
        1. Sort items by bradley_terry rating (descending)
        2. For each unassigned item in rank order:
           a. Start a new batch with this item
           b. Try to add next unassigned items that haven't battled any batch members
           c. Stop when batch reaches batch_size or no more valid candidates
           d. If batch >= min_batch_size, keep it; else return items to pool
        3. Return all valid batches
    """
    # Sort by current rating (highest first)
    sorted_df = headline_df.sort_values(
        'bradley_terry', ascending=False).reset_index(drop=True)

    # Track which items are assigned to batches
    assigned = set()
    batches = []

    for i, row in sorted_df.iterrows():
        story_id = int(row['id'])

        if story_id in assigned:
            continue

        # Start new batch with this item
        current_batch = [story_id]
        assigned.add(story_id)

        # Try to fill the batch
        for j in range(i + 1, len(sorted_df)):
            candidate_id = int(sorted_df.iloc[j]['id'])

            if candidate_id in assigned:
                continue

            # Check if candidate has battled any current batch member
            has_battled_any = False
            for batch_member_id in current_batch:
                if battle_history[candidate_id, batch_member_id] != 0:
                    has_battled_any = True
                    break

            # If no battles with any batch member, add to batch
            if not has_battled_any:
                current_batch.append(candidate_id)
                assigned.add(candidate_id)

                # Stop if we reached target batch size
                if len(current_batch) >= batch_size:
                    break

        # Keep last batch if it meets minimum size
        if len(current_batch) >= min_batch_size:
            batches.append(current_batch)
        else:
            # Return items to unassigned pool
            for item_id in current_batch:
                assigned.remove(item_id)

    # Mop-up phase: give unassigned stories random battles with ones they haven't battled
    # this ensures everyone gets a battle
    # otherwise the bottom stories have less chance to get out and there is hysteresis
    # compute unassigned stories
    # this also makes it return random batches when there are no unbattled pairs
    unassigned = set(sorted_df['id']) - assigned

    if unassigned:
        extra_battle_ids = []

        while unassigned:
            # Pick an unassigned story
            story_id = unassigned.pop()
            extra_battle_ids.append(story_id)

            # Find all unbattled opponents using numpy
            unbattled = set(
                np.where(battle_history[story_id] == 0)[0]) - {story_id}

            if unbattled:
                # Pick one random unbattled opponent
                candidate_id = np.random.choice(list(unbattled))
                extra_battle_ids.append(candidate_id)
                unassigned.discard(candidate_id)
            else:
                # if no unbattled, just pick a random opponent
                valid_ids = list(set(sorted_df['id']) - {story_id})
                candidate_id = np.random.choice(valid_ids)
                extra_battle_ids.append(candidate_id)
                unassigned.discard(candidate_id)

        # remove duplicates since we are adding without checking if already present
        extra_battle_ids = list(set(extra_battle_ids))

        # Use bt_paginate_list_async to create properly sized batches
        if len(extra_battle_ids) >= 1:  # edge case where 1 straggler and no unbattled
            async for batch in bt_paginate_list_async(extra_battle_ids, batch_size):
                batches.append(batch)

    return batches


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
                               batches: List[List[int]],
                               battle_agent,
                               max_concurrent=100,
                               logger=_logger) -> List[Tuple[int, int]]:
    """
    Process pre-formed battle batches concurrently and return winner/loser tuples.

    Args:
        bt_df: DataFrame with article data including 'id' and 'input_text' columns
        batches: List of batches, where each batch is a list of story IDs
        battle_agent: LLM agent for ranking battles
        max_concurrent: Maximum concurrent API calls
        logger: Logger instance

    Returns:
        List of (winner_id, loser_id) tuples from all battles
    """

    async def process_single_batch(batch_ids):
        """Process a single batch with semaphore control."""
        async with semaphore:
            try:
                # Get records for this batch in the order specified by batch_ids
                batch_records = []
                for story_id in batch_ids:
                    row = bt_df[bt_df['id'] == story_id].iloc[0]
                    batch_records.append({
                        'id': int(row['id']),
                        'input_text': row['input_text']
                    })

                result = await battle_agent.run_prompt(input_text=str(batch_records))
                return result
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                return None

    semaphore = asyncio.Semaphore(max_concurrent)

    logger.info(f"Processing {len(batches)} pre-formed batches")

    # Create tasks for each batch
    tasks = [process_single_batch(batch) for batch in batches]

    # Execute all battles concurrently
    logger.info(
        f"Processing {len(tasks)} battles with concurrency = {max_concurrent}")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Extract battle results
    retlist = []
    for result in results:
        if result is None or not hasattr(result, 'items'):
            continue

        battles = result.items
        n_results = len(battles)
        for i in range(n_results-1):
            for j in range(i+1, n_results):
                winner = battles[i].id
                loser = battles[j].id
                # logger.debug(f"Battle: {winner} beat {loser}")
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
    # but note that we don't get all stories into each round
    n_stories = len(bt_df)
    max_rounds = math.ceil((n_stories-1) / (BT_BATCH-1))
    logger.info(
        f"Max {max_rounds} rounds")
    # Battle history matrix (N x N)
    # matrix of battles, did 2 stories battle each other?
    battle_history = np.zeros((n_stories, n_stories), dtype=int)

    # Setup battle agent
    system, user, model = get_langfuse_client(
        logger=logger).get_prompt("newsagent/battle_prompt")
    battle_agent = LLMagent(
        system_prompt=system,
        user_prompt=user,
        model=model,
        reasoning_effort="low",
        output_type=StoryOrderList,
        verbose=False,
        logger=logger
    )
    all_battles = []
    all_results = []

    convergence_threshold = n_stories / 100
    logger.info(
        f"Convergence threshold: {convergence_threshold}")
    min_rounds = max_rounds // 2
    logger.info(
        f"Min rounds: {min_rounds}")

    # run rounds to ensure everyone battles at least once
    for round_num in range(1, max_rounds+1):
        logger.info(
            "---------------------------------------------------")
        logger.info(
            f"Running round {round_num} of max {max_rounds}")
        logger.info(
            "---------------------------------------------------")

        # Swiss batching - creates batches directly instead of pairs
        # Mop-up phase ensures all stories battle at least once per round
        batches = await swiss_batching(
            bt_df, battle_history, batch_size=BT_BATCH, min_batch_size=3)
        logger.info(f"Generated {len(batches)} battle batches")

        if not batches:
            logger.info("No more valid batches available")
            break

        # Log batch statistics
        total_stories_in_batches = sum(len(batch) for batch in batches)
        logger.info(
            f"Total stories in batches: {total_stories_in_batches} ; Total stories: {len(bt_df)}")

        # Process battles concurrently with pre-formed batches
        battle_results = await process_battle_round(bt_df,
                                                    batches,
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
        with pd.option_context('display.max_columns', None, 'display.width', None, 'display.max_colwidth', None):
            display(bt_df[['id', 'site_name', 'title', 'bradley_terry', 'bt_z']
                          ].sort_values('bradley_terry', ascending=False))

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
        if len(all_results) > min_rounds:  # do at least 1/2 of max rounds
            last_two = (all_results[-1] + all_results[-2]) / 2
            prev_two = (all_results[-3] + all_results[-4]) / 2
            logger.info(f"last_two: {last_two:.2f}, prev_two: {prev_two:.2f}")
            if (last_two) < convergence_threshold:
                logger.info("Convergence threshold achieved - stopping")
                break
            elif last_two < n_stories / 5 and last_two > prev_two:
                # stop if increasing but needs to be < 20% of stories
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

    # Quality assessment
    try:
        rating_df['low_quality'] = await run_prompt_on_dataframe(
            input_df=rating_df[['id', 'input_text']],
            prompt_name="newsagent/rate_quality",
            output_type=StoryRatings,
            value_field='low_quality',
            item_list_field='results_list',
            item_id_field='id',
            chunk_size=25,
            return_probabilities=True,
            logger=logger
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
        rating_df['on_topic'] = await run_prompt_on_dataframe(
            input_df=rating_df[['id', 'input_text']],
            prompt_name="newsagent/rate_on_topic",
            output_type=StoryRatings,
            value_field='on_topic',
            item_list_field='results_list',
            item_id_field='id',
            chunk_size=25,
            return_probabilities=True,
            logger=logger
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
        rating_df['important'] = await run_prompt_on_dataframe(
            input_df=rating_df[['id', 'input_text']],
            prompt_name="newsagent/rate_importance",
            output_type=StoryRatings,
            value_field='important',
            item_list_field='results_list',
            item_id_field='id',
            chunk_size=25,
            return_probabilities=True,
            logger=logger
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

    return rating_df
