from collections import Counter
import logging
import pandas as pd
import numpy as np
from typing import Optional

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan
import optuna
import umap

from openai import OpenAI
from pydantic import BaseModel, Field

from llm import paginate_df_async, LangfuseClient, LLMagent, get_langfuse_client


class TopicText(BaseModel):
    """Single article classification result for a canonical topic"""
    topic_text: str = Field(description="The desctiption of the topic")


MIN_COMPONENTS = 20
RANDOM_STATE = 42


def _create_extended_summary(row):
    """
    Create an extended summary by concatenating available article fields.

    Combines title, description, topics, and summary into a single text string
    for use in embedding generation and clustering analysis.

    Args:
        row: pandas Series or dict-like object containing article data with
             optional fields: 'title', 'description', 'topics', 'summary'

    Returns:
        str: Combined text summary with sections separated by double newlines.
             Empty string if no valid content is found.
    """
    parts = []

    # Add title if present
    if 'title' in row and row['title']:
        parts.append(str(row['title']).strip())

    # Add description if present
    if 'description' in row and row['description']:
        parts.append(str(row['description']).strip())

    # Add topics if present (join with commas)
    if 'topics' in row and row['topics']:
        if isinstance(row['topics'], list):
            topics_str = ", ".join(str(topic).strip()
                                   for topic in row['topics'] if topic)
        else:
            topics_str = str(row['topics']).strip()
        if topics_str:
            parts.append(topics_str)

    # Add summary if present
    if pd.notna(row.get('summary')) and row.get('summary'):
        parts.append(str(row['summary']).strip())

    return "\n\n".join(parts)


async def _get_embeddings_df(headline_data: pd.DataFrame, embedding_model: str = "text-embedding-3-large") -> pd.DataFrame:
    """
    Generate embeddings for article summaries using OpenAI's embedding API.

    Creates extended summaries from article data and generates vector embeddings
    in batches using the specified OpenAI embedding model. Returns a DataFrame
    with embeddings preserving the original article indices.

    Args:
        headline_data: DataFrame containing article data with fields like
                      'title', 'description', 'topics', 'summary'
        embedding_model: OpenAI embedding model name (default: "text-embedding-3-large")

    Returns:
        pd.DataFrame: DataFrame with embedding vectors as columns, indexed by
                     original article indices. Empty DataFrame if no valid summaries found.

    Raises:
        Exception: If OpenAI API calls fail or embedding generation encounters errors
    """

    # Create extended_summary column by concatenating available fields
    headline_data_copy = headline_data.copy()

    headline_data_copy['extended_summary'] = headline_data_copy.apply(
        _create_extended_summary, axis=1)

    # Filter to articles with non-empty extended summaries
    articles_with_summaries = headline_data_copy[
        (headline_data_copy['extended_summary'].notna()) &
        (headline_data_copy['extended_summary'] != '')
    ].copy()

    all_embeddings = []
    client = OpenAI()

    # Use paginate_df_async similar to do_dedupe.py
    async for batch_df in paginate_df_async(articles_with_summaries, 25):
        text_batch = batch_df["extended_summary"].to_list()
        response = client.embeddings.create(
            input=text_batch, model=embedding_model)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    # Create DataFrame with embeddings, preserving original index
    embedding_df = pd.DataFrame(
        all_embeddings,
        index=articles_with_summaries.index
    )

    return embedding_df


def calculate_clustering_metrics(embeddings_array, labels, clusterer=None):
    """
    Calculate comprehensive clustering quality metrics for HDBSCAN results.

    Computes various clustering evaluation metrics including silhouette score,
    Calinski-Harabasz index, Davies-Bouldin index, cluster size statistics,
    and HDBSCAN-specific metrics. Creates a composite score for optimization.

    Args:
        embeddings_array: numpy.ndarray of shape (n_samples, n_features)
                         Normalized embeddings used for clustering
        labels: numpy.ndarray of shape (n_samples,)
               Cluster labels from HDBSCAN (-1 indicates noise points)
        clusterer: hdbscan.HDBSCAN object, optional
                  HDBSCAN clusterer instance for accessing internal metrics

    Returns:
        dict: Dictionary containing clustering metrics:
              - 'n_clusters': Number of discovered clusters (excluding noise)
              - 'n_noise_points': Number of noise points (label -1)
              - 'noise_ratio': Fraction of points classified as noise
              - 'avg_cluster_size': Mean cluster size
              - 'std_cluster_size': Standard deviation of cluster sizes
              - 'min_cluster_size': Smallest cluster size
              - 'max_cluster_size': Largest cluster size
              - 'silhouette_score': Silhouette coefficient [-1, 1]
              - 'calinski_harabasz_score': Calinski-Harabasz index [0, inf)
              - 'davies_bouldin_score': Davies-Bouldin index [0, inf)
              - 'hdbscan_validity_index': HDBSCAN-specific validity measure
              - 'composite_score': Weighted combination of quality metrics
    """

    # Filter out noise points (-1 labels) for some metrics
    non_noise_mask = labels != -1
    non_noise_embeddings = embeddings_array[non_noise_mask]
    non_noise_labels = labels[non_noise_mask]

    metrics = {}

    # Basic cluster statistics
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)

    metrics['n_clusters'] = n_clusters
    metrics['n_noise_points'] = n_noise
    metrics['noise_ratio'] = n_noise / len(labels)

    # Cluster size distribution
    cluster_sizes = Counter(labels[labels != -1])
    if cluster_sizes:
        metrics['avg_cluster_size'] = np.mean(list(cluster_sizes.values()))
        metrics['std_cluster_size'] = np.std(list(cluster_sizes.values()))
        metrics['min_cluster_size'] = min(cluster_sizes.values())
        metrics['max_cluster_size'] = max(cluster_sizes.values())

    # Skip other metrics if we have too few clusters or too much noise
    if n_clusters < 2 or len(non_noise_labels) < 2:
        print("Warning: Too few clusters or too much noise for some metrics")
        return metrics

    # HDBSCAN-specific metrics
    # gives some divide by 0 errors
    if clusterer is not None:
        try:
            # Validity index (HDBSCAN's internal metric)
            validity_idx = hdbscan.validity.validity_index(
                embeddings_array, labels, metric='euclidean'
            )
            metrics['hdbscan_validity_index'] = validity_idx
        except Exception as e:
            print(f"Could not compute HDBSCAN validity index: {e}")

        # Cluster persistence (stability)
        if hasattr(clusterer, 'cluster_persistence_'):
            metrics['cluster_persistence'] = clusterer.cluster_persistence_

    # Scikit-learn clustering metrics (excluding noise points)
    try:
        # Silhouette Score (higher is better, range [-1, 1])
        sil_score = silhouette_score(
            non_noise_embeddings, non_noise_labels, metric='euclidean')
        metrics['silhouette_score'] = sil_score

        # Calinski-Harabasz Index (higher is better)
        ch_score = calinski_harabasz_score(
            non_noise_embeddings, non_noise_labels)
        metrics['calinski_harabasz_score'] = ch_score

        # Davies-Bouldin Index (lower is better)
        db_score = davies_bouldin_score(non_noise_embeddings, non_noise_labels)
        metrics['davies_bouldin_score'] = db_score

    except Exception as e:
        print(f"Could not compute sklearn metrics: {e}")

    # Custom composite score balancing cluster quality and quantity
    if 'silhouette_score' in metrics and n_clusters > 0:
        # Penalize too many small clusters or too few large clusters
        # Optimal around 10 clusters
        # cluster_balance = 1 / (1 + abs(np.log(n_clusters / 10)))
        # size_consistency = 1 / \
        # (1 + metrics.get('std_cluster_size', 0) /
        #  max(metrics.get('avg_cluster_size', 1), 1))
        # Penalize high noise
        # noise_penalty = 1 - min(metrics['noise_ratio'], 0.5)

        composite_score = (
            0.5 * max(metrics['silhouette_score'], 0) +  # Quality component
            0.5 * max(metrics['hdbscan_validity_index'], 0)
            #             0.1 * cluster_balance +                       # Quantity component
            #             0.1 * size_consistency +                      # Size consistency
            #             0.3 * noise_penalty                           # Noise penalty
        )
        metrics['composite_score'] = composite_score

    return metrics


def print_clustering_summary(metrics):
    """
    Print a formatted summary of clustering quality metrics.

    Displays clustering results in a human-readable format including
    cluster counts, noise statistics, size distributions, and quality scores.

    Args:
        metrics: dict containing clustering metrics from calculate_clustering_metrics()
                Keys should include 'n_clusters', 'noise_ratio', quality scores, etc.
    """
    print("=== Clustering Quality Metrics ===")
    print(f"Number of clusters: {metrics.get('n_clusters', 'N/A')}")
    print(
        f"Noise points: {metrics.get('n_noise_points', 'N/A')} ({metrics.get('noise_ratio', 0):.1%})")

    if 'avg_cluster_size' in metrics:
        print(
            f"Average cluster size: {metrics['avg_cluster_size']:.1f} ± {metrics.get('std_cluster_size', 0):.1f}")
        print(
            f"Cluster size range: {metrics.get('min_cluster_size', 'N/A')} - {metrics.get('max_cluster_size', 'N/A')}")

    print("=== Quality Scores ===")
    if 'silhouette_score' in metrics:
        print(
            f"Silhouette Score: {metrics['silhouette_score']:.3f} (higher is better)")
    if 'calinski_harabasz_score' in metrics:
        print(
            f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.1f} (higher is better)")
    if 'davies_bouldin_score' in metrics:
        print(
            f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f} (lower is better)")
    if 'hdbscan_validity_index' in metrics:
        print(
            f"HDBSCAN Validity Index: {metrics['hdbscan_validity_index']:.3f}")
    if 'composite_score' in metrics:
        print(
            f"Composite Score: {metrics['composite_score']:.3f} (higher is better)")
    print()


def objective(trial, embeddings_array):
    """
    Optuna objective function for optimizing HDBSCAN hyperparameters.

    Performs dimensionality reduction with TruncatedSVD followed by HDBSCAN
    clustering, evaluating the quality using a composite metric. Optimizes
    the number of SVD components, min_cluster_size, and min_samples parameters.

    Args:
        trial: optuna.Trial object for suggesting hyperparameter values
        embeddings_array: numpy.ndarray of shape (n_samples, n_features)
                         Normalized embedding vectors for clustering

    Returns:
        float: Negative composite score (since Optuna minimizes).
               Higher absolute values indicate better clustering quality.
               Returns -1.0 for invalid clustering results.
    """

    n_components = trial.suggest_int('n_components',
                                     MIN_COMPONENTS,
                                     embeddings_array.shape[1] // 4)

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    reduced_embeddings = svd.fit_transform(embeddings_array)
    # Re-normalize after SVD
    reduced_embeddings /= np.linalg.norm(reduced_embeddings,
                                         axis=1, keepdims=True)

    # HDBSCAN hyperparameters to optimize
    min_cluster_size = trial.suggest_int('min_cluster_size', 2, 10)
    min_samples = trial.suggest_int('min_samples', 2, min_cluster_size)

    # Fit HDBSCAN
    print("=== HDBSCAN Parameters ===")
    print(f"min_cluster_size:   {min_cluster_size}")
    print(f"min_samples:        {min_samples}")
    print(f"n_components:       {n_components}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    labels = clusterer.fit_predict(reduced_embeddings)

    # Calculate metrics
    metrics = calculate_clustering_metrics(
        reduced_embeddings, labels, clusterer)
    print_clustering_summary(metrics)

    # Return negative composite score (Optuna minimizes)
    composite_score = metrics.get('composite_score', -1.0)

    # Penalize if no valid clusters found or too much noise
    if metrics.get('n_clusters', 0) < 2 or metrics.get('noise_ratio', 1.0) > 0.8:
        composite_score = -1.0

    return -composite_score


def optimize_hdbscan(embeddings_array, n_trials=100, timeout=None):
    """
    Optimize HDBSCAN hyperparameters using Optuna Bayesian optimization.

    Performs automated hyperparameter tuning for HDBSCAN clustering including
    dimensionality reduction with TruncatedSVD. Uses TPE sampler and median
    pruner for efficient optimization with early stopping of poor trials.

    Args:
        embeddings_array: numpy.ndarray of shape (n_samples, n_features)
                         Normalized embedding vectors for clustering
        n_trials: int, default=100
                 Maximum number of optimization trials to run
        timeout: float or None, default=None
                Maximum optimization time in seconds (None for no time limit)

    Returns:
        dict: Optimization results containing:
              - 'study': optuna.Study object with complete optimization history
              - 'best_params': dict of optimal hyperparameters
              - 'best_score': float, best composite score achieved
              - 'best_clusterer': hdbscan.HDBSCAN object fitted with best parameters
              - 'best_labels': numpy.ndarray of cluster labels from best clustering
              - 'best_embeddings': numpy.ndarray of (possibly reduced) embeddings used
              - 'best_metrics': dict of clustering quality metrics for best result
              - 'svd_transformer': TruncatedSVD object if dimensionality reduction applied
    """

    print(f"Starting optimization with {n_trials} trials...")
    print(f"Original embedding shape: {embeddings_array.shape}")

    # Create study
    study = optuna.create_study(
        direction='minimize',  # We return negative composite score
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, embeddings_array),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    # Get best parameters
    best_params = study.best_params
    best_score = -study.best_value  # Convert back to positive

    print("\nOptimization completed!")
    print(f"Best composite score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")

    # Test best parameters
    print("\n=== Results with Best Parameters ===")

    # Apply best dimensionality reduction
    if best_params['n_components'] < embeddings_array.shape[1]:
        reducer = TruncatedSVD(
            n_components=best_params['n_components'], random_state=RANDOM_STATE)
        best_embeddings = reducer.fit_transform(embeddings_array)
        # Re-normalize after SVD
        best_embeddings /= np.linalg.norm(embeddings_array,
                                          axis=1, keepdims=True)
        # reducer = umap.UMAP(n_components=best_params['n_components'])
        # # Fit the reducer to the data without transforming
        # reducer.fit(embeddings_array)
        # force np64 or hdbscan pukes
        # best_embeddings = reducer.transform(
        #     embeddings_array).astype(np.float64)
        # # Re-normalize after UMAP
        # best_embeddings /= np.linalg.norm(best_embeddings,
        #                                   axis=1, keepdims=True)

        print(
            f"Reduced dimensions from {embeddings_array.shape[1]} to {best_params['n_components']}")
    else:
        best_embeddings = embeddings_array
        # reducer = None
        print("No dimensionality reduction applied")

    # Fit with best parameters
    best_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=best_params['min_cluster_size'],
        min_samples=best_params['min_samples'],
        metric="euclidean",
        cluster_selection_method="eom",
    )

    best_labels = best_clusterer.fit_predict(best_embeddings)
    best_metrics = calculate_clustering_metrics(
        best_embeddings, best_labels, best_clusterer)

    print_clustering_summary(best_metrics)
    print()

    # Return results
    return {
        'study': study,
        'best_params': best_params,
        'best_score': best_score,
        'best_clusterer': best_clusterer,
        'best_labels': best_labels,
        'best_embeddings': best_embeddings,
        'best_metrics': best_metrics,
        'svd_transformer': reducer if best_params['n_components'] < embeddings_array.shape[1] else None
    }


async def name_clusters(headline_df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Generate descriptive names for discovered clusters using AI.

    Uses an LLM to analyze article titles and topics within each cluster
    to generate meaningful cluster names. Clusters with fewer than 2 articles
    or noise clusters (id < 0) are labeled as "Other".

    Args:
        headline_df: pd.DataFrame containing articles with 'cluster', 'title',
                    and 'topics' columns. Cluster column should contain integer
                    cluster IDs from HDBSCAN (-1 for noise points).
        logger: logging.Logger for recording cluster naming progress and errors

    Returns:
        pd.DataFrame: Input DataFrame with additional 'cluster_name' column
                     containing descriptive names for each cluster.
                     Articles in noise/small clusters get "Other" as cluster_name.

    Raises:
        Exception: If LLM prompting fails or cluster analysis encounters errors.
                  Individual cluster naming failures are logged but don't stop processing.
    """

    system_prompt, user_prompt, model, reasoning_effort = get_langfuse_client(
        logger=logger).get_prompt("newsagent/topic_writer")

    topic_writer = LLMagent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_type=TopicText,
        model=model,
        reasoning_effort=reasoning_effort,
        verbose=False,
        logger=logger
    )

    cluster_list = [(len(headline_df.loc[headline_df["cluster"] == i]), int(i))
                    for i in headline_df['cluster'].unique()]
    cluster_list.sort(reverse=True)

    # default to "Other"
    headline_df["cluster_name"] = 'Other'

    # for each cluster, prompt for a name and update cluster_name in headline_df
    cluster_topics = [list() for _ in range(len(cluster_list)-1)]
    for n, i in cluster_list:
        try:
            if i < 0 or n < 2:
                continue
            tmpdf = headline_df.loc[headline_df['cluster'] == i].copy()
            titles = tmpdf["title"].to_list()
            topics = tmpdf["topics"].to_list()
            title_topics = [
                f'{title} ({", ".join(topic_list)})' for title, topic_list in zip(titles, topics)]
            topic_text = await topic_writer.run_prompt(input_text=str(title_topics))
            cluster_topics[i] = topic_text.topic_text
            headline_df.loc[headline_df['cluster'] == i,
                            "cluster_name"] = topic_text.topic_text
            if logger:
                logger.info(f"{i}: {topic_text.topic_text}")
                logger.info("\n".join(title_topics))
                logger.info("\n")
        except Exception as exc:
            if logger:
                logger.error(exc)
    return headline_df


async def do_clustering(headline_df: pd.DataFrame, logger: Optional[logging.Logger] = None,
                        embedding_model: str = "text-embedding-3-large") -> pd.DataFrame:
    """
    Perform complete clustering workflow on article headlines.

    Executes the full clustering pipeline: creates extended summaries from article data,
    generates embeddings using OpenAI's API, optimizes HDBSCAN hyperparameters with
    Optuna, assigns cluster labels, and generates descriptive cluster names using AI.

    Args:
        headline_df: pd.DataFrame containing article data with columns:
                    'title', 'description', 'topics', 'summary' (at minimum)
        logger: logging.Logger for tracking clustering progress and errors
        embedding_model: str, default="text-embedding-3-large"
                        OpenAI embedding model name for vector generation

    Returns:
        pd.DataFrame: Input DataFrame with additional columns:
                     - 'extended_summary': Combined text from multiple fields
                     - 'cluster': Integer cluster ID (-1 for noise points)
                     - 'cluster_name': Descriptive name for each cluster

    Raises:
        Exception: If embedding generation, optimization, or cluster naming fails.
                  Individual failures in cluster naming are handled gracefully.

    Note:
        Uses 200 optimization trials for hyperparameter tuning, which may take
        several minutes depending on the dataset size and hardware.
    """

    headline_df['extended_summary'] = headline_df.apply(
        _create_extended_summary, axis=1)
    embeddings_df = await _get_embeddings_df(pd.DataFrame(headline_df))
    results = optimize_hdbscan(embeddings_df, n_trials=200)

    best_labels = results['best_labels']
    cluster_df = pd.DataFrame(
        {'cluster': best_labels}, index=embeddings_df.index)
    headline_df['cluster'] = cluster_df['cluster'].astype(int)

    headline_df = await name_clusters(headline_df, logger)
    return headline_df
