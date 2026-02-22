import choix
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import logging
import asyncio
from pydantic import BaseModel, Field
from IPython.display import display
from scipy.stats import zscore
from llm import LLMagent, run_prompt_on_dataframe
from prompts import load_prompt

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


class StoryConfidence(BaseModel):
    """StoryConfidence class for structured output confidence scoring"""
    id: int = Field(description="The id of the story")
    confidence: float = Field(description="Confidence score 0.0 to 1.0")


class StoryConfidenceList(BaseModel):
    """List of StoryConfidence for structured output"""
    results_list: list[StoryConfidence]


class StoryOrder(BaseModel):
    """StoryOrder class for generic structured output rating"""
    id: int = Field(description="The id of the story")


class StoryOrderList(BaseModel):
    """List of StoryOrder for structured output"""
    items: List[StoryOrder] = Field(
        description="List of StoryOrder")


def swiss_pairing(headline_df: pd.DataFrame, battle_history: set) -> List[Tuple[int, int]]:
    """
    Create Swiss-style pairings for Bradley-Terry battles.
    Returns all valid pairs based on current ranking.

    Args:
        headline_df: DataFrame sorted by current rating (best first)
        battle_history: Set of (id_a, id_b) tuples already battled

    Returns:
        List of (story1_id, story2_id) pairs for this round
    """
    used_this_round = set()
    pairs = []

    # Sort by current rating (highest first)
    sorted_df = headline_df.sort_values(
        'bradley_terry', ascending=False).reset_index(drop=True)

    for i, row in sorted_df.iterrows():
        aid = int(row['id'])
        if aid in used_this_round:
            continue

        # Find best available opponent (highest rated, not yet battled)
        for j in range(int(i) + 1, len(sorted_df)):
            bid = int(sorted_df.iloc[j]['id'])
            if bid in used_this_round:
                continue
            # Skip if already battled (check both orderings)
            if (aid, bid) in battle_history or (bid, aid) in battle_history:
                continue

            pairs.append((aid, bid))
            used_this_round.add(aid)
            used_this_round.add(bid)
            break

    return pairs


async def swiss_batching(headline_df: pd.DataFrame, battle_history: set,
                         batch_size: int = BT_BATCH) -> List[List[dict]]:
    """
    Get pairs from swiss_pairing and group into batches.

    Each batch is a list of dicts with 'id', 'title', 'summary' — ready
    to be sent to the battle LLM.

    Args:
        headline_df: DataFrame with 'id', 'title', 'summary', 'bradley_terry'.
        battle_history: Set of (id_a, id_b) tuples already battled.
        batch_size: Max items per batch.

    Returns:
        List of batches (each batch is a list of article dicts).
    """
    pairs = swiss_pairing(headline_df, battle_history)
    if not pairs:
        return []

    # Flatten all unique IDs from pairs, preserving order
    all_ids: List[int] = []
    for a, b in pairs:
        if a not in all_ids:
            all_ids.append(a)
        if b not in all_ids:
            all_ids.append(b)

    # Build lookup of article data
    id_to_row = {}
    for _, row in headline_df.iterrows():
        id_to_row[int(row["id"])] = {
            "id": int(row["id"]),
            "title": str(row.get("title", "")),
            "summary": str(row.get("summary", "")),
        }

    # Group IDs into batches of batch_size
    batches: List[List[dict]] = []
    for i in range(0, len(all_ids), batch_size):
        chunk_ids = all_ids[i: i + batch_size]
        batch = [id_to_row[cid] for cid in chunk_ids if cid in id_to_row]
        if len(batch) >= 2:
            batches.append(batch)

    return batches


async def process_battle_round(batches: List[List[dict]],
                               battle_agent,
                               max_concurrent=100,
                               logger=_logger) -> List[Tuple[int, int]]:
    """
    Process pre-formed battle batches concurrently and return winner/loser tuples.

    Args:
        batches: List of batches, where each batch is a list of article dicts
                 (each dict has 'id', 'title', 'summary')
        battle_agent: LLM agent for ranking battles
        max_concurrent: Maximum concurrent API calls
        logger: Logger instance

    Returns:
        List of (winner_id, loser_id) tuples from all battles
    """

    async def process_single_batch(batch: List[dict]):
        """Process a single batch with semaphore control."""
        async with semaphore:
            try:
                result = await battle_agent.run_prompt(input_text=str(batch))
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
        for i in range(n_results - 1):
            for j in range(i + 1, n_results):
                winner = battles[i].id
                loser = battles[j].id
                retlist.append((winner, loser))

    return retlist


async def bradley_terry(headline_df: pd.DataFrame, logger=_logger) -> pd.DataFrame:
    """
    Enhanced Bradley-Terry rating with Swiss pairing and async battles.

    Args:
        headline_df: DataFrame containing articles to be ranked.
        logger: Logger instance

    Returns:
        DataFrame with additional 'bradley_terry' and 'bt_z' columns.
    """
    logger.info("Running Bradley-Terry rating with Swiss pairing")

    bt_df = headline_df.copy()
    n = len(bt_df)

    if n < 3:
        bt_df['bradley_terry'] = 0.0
        bt_df['bt_z'] = 0.0
        return bt_df

    # Build ID <-> contiguous index mapping (choix needs 0-based indices)
    ids = bt_df["id"].tolist()
    id_to_idx = {int(aid): idx for idx, aid in enumerate(ids)}

    # Initial rating by position
    bt_df['bradley_terry'] = np.linspace(1.0, 0.0, n)

    max_rounds = 8
    logger.info(f"Max {max_rounds} rounds")

    battle_history: set = set()
    all_battles: List[Tuple[int, int]] = []  # (winner_idx, loser_idx) for choix

    # Setup battle agent
    system, user, model, reasoning_effort = load_prompt(
        "newsagent/battle_prompt")
    battle_agent = LLMagent(
        system_prompt=system,
        user_prompt=user,
        model=model,
        reasoning_effort=reasoning_effort,
        output_type=StoryOrderList,
        verbose=False,
        logger=logger
    )

    convergence_threshold = max(1, n * 0.005)
    logger.info(f"Convergence threshold: {convergence_threshold}")
    min_rounds = max_rounds // 2
    logger.info(f"Min rounds: {min_rounds}")

    previous_rankings = bt_df.sort_values(
        "bradley_terry", ascending=False)["id"].values.copy()
    all_results: List[float] = []

    for round_num in range(1, max_rounds + 1):
        logger.info(
            "---------------------------------------------------")
        logger.info(
            f"Running round {round_num} of max {max_rounds}")
        logger.info(
            "---------------------------------------------------")

        # Swiss batching
        batches = await swiss_batching(
            bt_df, battle_history, batch_size=BT_BATCH)
        logger.info(f"Generated {len(batches)} battle batches")

        if not batches:
            logger.info("No more valid batches available")
            break

        # Log batch statistics
        total_stories_in_batches = sum(len(batch) for batch in batches)
        logger.info(
            f"Total stories in batches: {total_stories_in_batches} ; Total stories: {n}")

        # Process battles concurrently with pre-formed batches
        battle_results = await process_battle_round(
            batches, battle_agent, max_concurrent=1000, logger=logger)

        # Update battle history and accumulate results
        for winner_id, loser_id in battle_results:
            battle_history.add((winner_id, loser_id))
            battle_history.add((loser_id, winner_id))
            # Map to contiguous indices for choix
            w_idx = id_to_idx.get(winner_id)
            l_idx = id_to_idx.get(loser_id)
            if w_idx is not None and l_idx is not None:
                all_battles.append((w_idx, l_idx))

        logger.info(f"total battles: {len(all_battles)}")

        if not all_battles:
            continue

        # Compute BT parameters using choix with contiguous indices
        logger.info("Recomputing Bradley-Terry ratings")
        bt_df['bradley_terry'] = choix.opt_pairwise(n, all_battles)
        logger.info("Recomputed Bradley-Terry ratings")

        # Z-score normalize
        bt_values = bt_df["bradley_terry"].values
        if np.std(bt_values) > 0:
            bt_df["bt_z"] = zscore(bt_values, ddof=0)
        else:
            bt_df["bt_z"] = np.zeros(n)
        logger.info("Computed Bradley-Terry z-scores")

        with pd.option_context('display.max_columns', None, 'display.width', None, 'display.max_colwidth', None):
            display(bt_df[['id', 'site_name', 'title', 'bradley_terry', 'bt_z']
                          ].sort_values('bradley_terry', ascending=False))

        # Check convergence using positional displacement
        new_rankings = bt_df.sort_values(
            "bradley_terry", ascending=False)["id"].values
        ranking_change_sum = np.abs(
            np.array([np.where(new_rankings == pid)[0][0] for pid in ids])
            - np.array([np.where(previous_rankings == pid)[0][0] for pid in ids])
        ).sum()
        avg_change = ranking_change_sum / n
        previous_rankings = new_rankings.copy()

        logger.info(f"After round {round_num}/{max_rounds}: ")
        logger.info(
            f"Sum of absolute ranking changes: {ranking_change_sum:.1f} (avg rank chg {avg_change:.2f})")

        all_results.append(avg_change)

        # Convergence detection
        if len(all_results) > min_rounds:  # do at least 1/2 of max rounds
            last_two = (all_results[-1] + all_results[-2]) / 2
            if last_two < convergence_threshold:
                logger.info("Convergence threshold achieved - stopping")
                break
            if len(all_results) >= 4:
                prev_two = (all_results[-3] + all_results[-4]) / 2
                logger.info(f"last_two: {last_two:.2f}, prev_two: {prev_two:.2f}")
                if last_two < n / 5 and last_two > prev_two:
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

    # Run LLM assessments concurrently
    if logger:
        logger.info("Rating quality, topic relevance, and importance concurrently")

    lq_task = run_prompt_on_dataframe(
        input_df=rating_df[['id', 'input_text']],
        prompt_name="newsagent/rate_quality",
        output_type=StoryConfidenceList,
        value_field='confidence',
        item_list_field='results_list',
        item_id_field='id',
        chunk_size=25,
        logger=logger
    )
    ot_task = run_prompt_on_dataframe(
        input_df=rating_df[['id', 'input_text']],
        prompt_name="newsagent/rate_on_topic",
        output_type=StoryConfidenceList,
        value_field='confidence',
        item_list_field='results_list',
        item_id_field='id',
        chunk_size=25,
        logger=logger
    )
    imp_task = run_prompt_on_dataframe(
        input_df=rating_df[['id', 'input_text']],
        prompt_name="newsagent/rate_importance",
        output_type=StoryConfidenceList,
        value_field='confidence',
        item_list_field='results_list',
        item_id_field='id',
        chunk_size=25,
        logger=logger
    )

    results = await asyncio.gather(lq_task, ot_task, imp_task, return_exceptions=True)

    if isinstance(results[0], Exception):
        if logger:
            logger.warning(f"Failed to get quality assessment: {results[0]}")
        rating_df['low_quality'] = 0
    else:
        rating_df['low_quality'] = results[0]

    if isinstance(results[1], Exception):
        if logger:
            logger.warning(f"Failed to get topic relevance assessment: {results[1]}")
        rating_df['on_topic'] = 1
    else:
        rating_df['on_topic'] = results[1]

    if isinstance(results[2], Exception):
        if logger:
            logger.warning(f"Failed to get importance assessment: {results[2]}")
        rating_df['important'] = 1
    else:
        rating_df['important'] = results[2]

    if logger:
        logger.info(f"low quality: {rating_df['low_quality'].value_counts().to_dict()}")
        logger.info(f"on topic: {rating_df['on_topic'].value_counts().to_dict()}")
        logger.info(f"important: {rating_df['important'].value_counts().to_dict()}")

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
