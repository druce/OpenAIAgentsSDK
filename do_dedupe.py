"""
Dedupe by cosine similarity
If a popular article AP or Reuters article is syndicated by multiple sources,
near-identical text will show up as different URLs
"""
from llm import paginate_df_async
import pandas as pd
import numpy as np
import pickle
import os
import logging
import hdbscan
from collections import Counter
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import List, Any, Tuple, Dict, Optional
from scrape import trunc_tokens
# from config import MAX_TOKENS
MAX_TOKENS = 8192
SIMILARITY_THRESHOLD = 0.925


# def read_and_truncate_files(text_paths: List[str]) -> List[str]:
#     """Read files and truncate their contents."""
#     truncated_texts = []

#     for path in text_paths:
#         try:
#             with open(path, 'r', encoding='utf-8') as file:
#                 content = file.read()
#                 truncated_content = trunc_tokens(content)
#                 truncated_texts.append(truncated_content)
#         except Exception as e:
#             print(f"Error reading {path}: {e}")
#             truncated_texts.append("")  # Empty string for failed reads

#     return truncated_texts


def read_and_truncate_files(df: pd.DataFrame, model: str = 'gpt-4o', max_tokens: int = MAX_TOKENS) -> pd.DataFrame:
    """Read files and truncate their contents using tiktoken."""
    ret_df = df.copy()
    # drop rows where text_path is NaN
    ret_df = ret_df.dropna(subset=['text_path'])
    # drop rows where text_path is 'download/text/.txt'
    ret_df = ret_df[ret_df['text_path'] != 'download/text/.txt']

    truncated_texts = []
    for row in ret_df.itertuples():
        path = row.text_path
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
                truncated_content = trunc_tokens(
                    content, model=model, maxtokens=max_tokens)
                truncated_texts.append(truncated_content)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            truncated_texts.append("")  # Empty string for failed reads

    ret_df['truncated_text'] = truncated_texts
    ret_df = ret_df.loc[ret_df['truncated_text'].str.len() > 0]

    return ret_df


async def get_embeddings_batch(df: pd.DataFrame, embedding_model: str = "text-embedding-3-large") -> List[List[float]]:
    """
    Get embeddings for a list of texts using OpenAI client with pagination.
    """

    all_embeddings = []
    client = OpenAI()
    async for batch_df in paginate_df_async(df, 25):
        text_batch = batch_df["text_path"].to_list()
        response = client.embeddings.create(
            input=text_batch, model=embedding_model)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def create_indexed_similarity_matrix(embeddings: List[List[float]], original_index: pd.Index) -> pd.DataFrame:
    """Create cosine similarity matrix with proper indexing."""
    embeddings_array = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings_array)

    # Convert to DataFrame with proper indexing
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=original_index,
        columns=original_index
    )

    return similarity_df


def find_high_similarity_pairs(similarity_df: pd.DataFrame, threshold: float = 0.95) -> List[Tuple]:
    """Find pairs of indices with similarity above threshold."""
    high_similarity_pairs = []

    # Get upper triangle (excluding diagonal) to avoid duplicates
    for i in range(len(similarity_df)):
        for j in range(i + 1, len(similarity_df)):
            if similarity_df.iloc[i, j] > threshold:
                high_similarity_pairs.append(
                    (similarity_df.index[i], similarity_df.index[j]))

    return high_similarity_pairs


def filter_similar_rows(df: pd.DataFrame, high_similarity_pairs: List[Tuple]) -> pd.DataFrame:
    """
    Filter out rows that have high similarity with other rows.
    Keeps the first occurrence in each similar pair.
    """
    indices_to_remove = set()

    for idx1, idx2 in high_similarity_pairs:
        # Keep the one with higher content_length
        if df.loc[idx1, 'content_length'] > df.loc[idx2, 'content_length']:
            indices_to_remove.add(idx2)
        else:
            indices_to_remove.add(idx1)

        print(f"  Pair: {idx1} vs {idx2}")
        print(
            f"    {idx1}: {df.loc[idx1, 'source']} - {df.loc[idx1, 'title']}")
        print(
            f"    {idx1}: {df.loc[idx1, 'final_url']}")
        print(
            f"    {idx2}: {df.loc[idx2, 'source']} - {df.loc[idx2, 'title']}")
        print(
            f"    {idx2}: {df.loc[idx2, 'final_url']}")

    print(f"Removing {len(indices_to_remove)} rows due to high similarity ")

    # Filter the dataframe
    filtered_df = df.drop(indices_to_remove)

    return filtered_df


async def process_dataframe_with_filtering(df: pd.DataFrame, similarity_threshold: float = SIMILARITY_THRESHOLD,
                                           embedding_model: str = 'text-embedding-3-large') -> pd.DataFrame:

    print(f"Starting with {len(df)} rows...")

    # Step 1: Make a list of all text_paths in column order
    text_paths = df['text_path'].fillna('')
    print(f"Processing {len(text_paths)} files...")

    # Step 2: Read and truncate file contents using tiktoken
    print(
        f"Reading and truncating files to {MAX_TOKENS} tokens using {embedding_model} tokenizer...")
    truncated_texts = read_and_truncate_files(
        df, model=embedding_model, max_tokens=MAX_TOKENS)

    # Step 3: Get embeddings using OpenAI client
    print(f"Getting embeddings for {len(truncated_texts)} texts...")
    embeddings = await get_embeddings_batch(truncated_texts)

    # Step 4: Create indexed cosine similarity matrix
    print("Creating indexed similarity matrix...")
    similarity_df = create_indexed_similarity_matrix(
        embeddings, truncated_texts.index)

    # Step 5: Find high similarity pairs
    print(f"Finding pairs with similarity > {similarity_threshold}...")
    high_similarity_pairs = find_high_similarity_pairs(
        similarity_df, similarity_threshold)

    # Step 6: Filter the original dataframe
    print("Filtering dataframe...")
    filtered_df = filter_similar_rows(df, high_similarity_pairs)
    print(
        f"Final dataframe has {len(filtered_df)} rows (removed {len(df) - len(filtered_df)} rows)")

    return filtered_df

# Usage example:


def main():
    # Initialize OpenAI client
    # or use environment variable OPENAI_API_KEY
    client = OpenAI(api_key="your-api-key-here")

    # Load your dataframe
    df = pd.read_csv('your_data.csv')  # or however you load your dataframe

    # Process the dataframe with filtering
    filtered_df, similarity_matrix = process_dataframe_with_filtering(
        df,
        client,
        similarity_threshold=0.95,
        # Can change to 'gpt-3.5-turbo', 'text-davinci-003', etc.
        tokenizer_model='gpt-4o',
        max_tokens=8192,
        # or 'text-embedding-3-small', 'text-embedding-ada-002'
        embedding_model='text-embedding-3-large'
    )

    print(f"\nOriginal dataframe shape: {df.shape}")
    print(f"Filtered dataframe shape: {filtered_df.shape}")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    # Show some high similarity pairs if any exist
    high_sim_pairs = find_high_similarity_pairs(similarity_matrix, 0.95)
    if high_sim_pairs:
        print(f"\nFound {len(high_sim_pairs)} high similarity pairs:")
        for pair in high_sim_pairs[:5]:  # Show first 5
            sim_score = similarity_matrix.loc[pair[0], pair[1]]
            print(
                f"  Indices {pair[0]} - {pair[1]}: similarity = {sim_score:.4f}")

    # Optionally save results
    # filtered_df.to_csv('filtered_dataframe.csv', index=True)
    # similarity_matrix.to_csv('similarity_matrix.csv')


def make_bullet(row):
    """Create a bullet point from article data"""
    return f"• {row.get('title', '')}: {row.get('summary', '')}"


def nearest_neighbor_sort(embedding_df: pd.DataFrame) -> List[int]:
    """Sort embeddings using nearest neighbor greedy approach"""
    embeddings = embedding_df.values
    n = len(embeddings)

    if n <= 1:
        return list(range(n))

    # Start with first point
    unvisited = set(range(1, n))
    path = [0]
    current = 0

    while unvisited:
        # Find nearest unvisited point
        distances = [np.linalg.norm(
            embeddings[current] - embeddings[i]) for i in unvisited]
        nearest_idx = min(range(len(distances)), key=distances.__getitem__)
        nearest = list(unvisited)[nearest_idx]

        path.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    return path


def calculate_clustering_metrics(embeddings_array: np.ndarray, labels: np.ndarray, clusterer: Optional[Any] = None) -> Dict[str, Any]:
    """
    Calculate various clustering quality metrics for HDBSCAN results.

    Args:
        embeddings_array: Original normalized embeddings used for clustering
        labels: Cluster labels from HDBSCAN
        clusterer: Optional HDBSCAN clusterer object

    Returns:
        Dictionary of clustering metrics
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
    if clusterer is not None:
        try:
            # Validity index (HDBSCAN's internal metric)
            validity_idx = hdbscan.validity.validity_index(
                embeddings_array, labels, metric='euclidean'
            )
            metrics['hdbscan_validity_index'] = validity_idx
        except (ValueError, ZeroDivisionError, RuntimeError) as e:
            print(f"Could not compute HDBSCAN validity index: {e}")
        except ImportError as e:
            print(f"HDBSCAN validity module not available: {e}")

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

    except ValueError as e:
        print(
            f"Invalid data for sklearn metrics (likely insufficient clusters): {e}")
    except RuntimeError as e:
        print(f"Runtime error computing sklearn metrics: {e}")

    # Custom composite score balancing cluster quality and quantity
    if 'silhouette_score' in metrics and n_clusters > 0:
        composite_score = (
            0.5 * max(metrics['silhouette_score'], 0) +  # Quality component
            0.5 * max(metrics.get('hdbscan_validity_index', 0), 0)
        )
        metrics['composite_score'] = composite_score

    return metrics


def print_clustering_summary(metrics: Dict[str, Any]):
    """Print a summary of clustering metrics"""
    print("\n=== Clustering Summary ===")
    print(f"Number of clusters: {metrics.get('n_clusters', 'N/A')}")
    print(
        f"Noise points: {metrics.get('n_noise_points', 'N/A')} ({metrics.get('noise_ratio', 0):.1%})")

    if 'avg_cluster_size' in metrics:
        print(f"Average cluster size: {metrics['avg_cluster_size']:.1f}")
        print(
            f"Cluster size range: {metrics.get('min_cluster_size', 'N/A')} - {metrics.get('max_cluster_size', 'N/A')}")

    if 'silhouette_score' in metrics:
        print(f"Silhouette score: {metrics['silhouette_score']:.3f}")

    if 'composite_score' in metrics:
        print(f"Composite score: {metrics['composite_score']:.3f}")


def cluster_summaries(aidf: pd.DataFrame, data_root: str = ".",
                      embedding_model: str = 'text-embedding-3-large',
                      logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Cluster news summaries using OpenAI embeddings and saved UMAP dimensionality reduction model.

    Args:
        aidf: DataFrame with articles containing summary column
        data_root: Root directory containing umap_reducer.pkl
        embedding_model: OpenAI embedding model to use
        logger: Optional logger for progress messages

    Returns:
        DataFrame with cluster assignments and sorting
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    log(f"Fetching embeddings for {len(aidf)} headlines")
    client = OpenAI()

    # Create bullet points for embedding
    aidf = aidf.copy()
    aidf["bullet"] = aidf.apply(make_bullet, axis=1)

    # Get bullet embeddings
    response = client.embeddings.create(
        input=aidf['bullet'].tolist(),
        model=embedding_model
    )
    embedding_df = pd.DataFrame(
        [e.model_dump()['embedding'] for e in response.data]
    )

    # Greedy traveling salesman sort
    log("Sort with nearest_neighbor_sort")
    sorted_indices = nearest_neighbor_sort(embedding_df)
    aidf['sort_order'] = sorted_indices

    # Load UMAP dimensionality reduction model
    log("Load UMAP dimensionality reduction model")
    umap_path = os.path.join(data_root, "umap_reducer.pkl")

    if not os.path.exists(umap_path):
        log(
            f"Warning: UMAP model not found at {umap_path}. Skipping dimensionality reduction.")
        reduced_data = embedding_df.values.astype(np.float64)
    else:
        with open(umap_path, 'rb') as pklfile:
            reducer = pickle.load(pklfile)

        log("Perform dimensionality reduction")
        # Force np64 or hdbscan complains
        reduced_data = reducer.transform(
            embedding_df.values).astype(np.float64)
        # Renormalize after dimensionality reduction
        reduced_data /= np.linalg.norm(reduced_data, axis=1, keepdims=True)

    # Use HDBSCAN with best params
    log("Cluster with HDBSCAN")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    labels = clusterer.fit_predict(reduced_data)
    aidf['cluster'] = labels

    # Calculate metrics
    metrics = calculate_clustering_metrics(reduced_data, labels, clusterer)
    print_clustering_summary(metrics)

    log(f"Found {len(aidf['cluster'].unique())-1} clusters")

    # Assign unclustered items to cluster 999
    aidf.loc[aidf['cluster'] == -1, 'cluster'] = 999

    # Sort first by clusters found by HDBSCAN, then by semantic ordering
    aidf = aidf.sort_values(['cluster', 'sort_order']).reset_index(drop=True)
    aidf = aidf.reset_index().drop(
        columns=["id"] if "id" in aidf.columns else [])
    aidf = aidf.rename(columns={'index': 'id'})

    # Initialize cluster names
    aidf["cluster_name"] = ""

    return aidf


if __name__ == "__main__":
    main()
