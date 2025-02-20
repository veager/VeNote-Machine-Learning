import copy
from typing import Dict, Optional, Any

import tqdm
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from .models import compute_combined_centroid, compute_combined_distance, compute_distance_matrix
from sklearn.metrics import silhouette_score


def combined_within_cluster_distance(
    data,
    labels : np.ndarray,
    centroids : np.ndarray,
    alpha : Optional[float] = 0.5,
    squared : Optional[bool] = False,
    n_jobs  : Optional[int] = 1,
    scaler_norm : Optional[float|None] = None,
    scaler_dtw : Optional[float|None] = None,
    norm_order : Optional[Dict|None] = None,
    dtw_kwargs : Optional[Dict|None] = None
):
    """Calculate total within-cluster distances using combined metric.

    Parameters:
    -----------
    X (np.ndarray):
        Input data, shape of (n_samples, n_timesteps, n_features) or (n_samples, n_timesteps)
    labels (np.ndarray):
        Cluster labels, shape of (n_samples, )
    centroids (np.ndarray):
        Cluster centroids, shape of (n_clusters, n_timesteps, n_features) or (n_clusters, n_timesteps)
    alpha (float):
        Weight for norm distance (0-1)
    n_jobs (int):
        Number of parallel jobs.
    scaler_norm (float, optional):
        Scaling factor for norm distance
    scaler_dtw (float, optional):
        Scaling factor for DTW distance
    norm_order (float, optional):
        Order for norm calculation
    dtw_kwargs (dict, optional):
        Additional DTW parameters

    Returns:
    --------
    np.ndarray: Array of within-cluster distances (shape: [n_clusters])
    """
    if len(data) != len(labels):
        raise ValueError(f"Data and labels must have the same length, "
                         f"got data shape : {data.shape}, labels shape : {labels.shape}")
    if len(centroids) != len(np.unique(labels)):
        raise ValueError(f"Centroid count must match unique label count, "
                         f"got centroid shape : {centroids.shape}, unique labels: {np.unique(labels)}")

    n_clusters = len(centroids)

    # Index pair from sample idx to its label idx
    cluster_indices = [(i, l) for i, l in enumerate(labels.tolist())]

    # Compute distance from each sample to its centroid
    dist = Parallel(n_jobs=n_jobs)(
        delayed(compute_combined_distance)(data[i], centroids[l], alpha, scaler_norm, scaler_dtw, norm_order, dtw_kwargs)
        for i, l in cluster_indices
    )

    # squared distance
    if squared:
        dist = [d*d for d in dist]

    # dist_sum = [sum([dist[i] for i, l in idx_pair if l == k]) for k in range(n_clusters)]
    cluster_sums = np.zeros(n_clusters)
    for (i, label), _dist_2 in zip(cluster_indices , dist):
        cluster_sums[label] += _dist_2
    return cluster_sums
# ==============================================================================================================
def combined_calinski_harabasz_score(
    X : np.ndarray,
    labels : np.ndarray,
    centroids : np.ndarray,
    alpha: float = 0.5,
    n_jobs: int = 1,
    scaler_norm: Optional[float] = None,
    scaler_dtw: Optional[float] = None,
    norm_order: Optional[float] = None,
    dtw_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute Calinski-Harabasz Score using the combined distance metric.
    """
    n_samples, n_clusters = X.shape[0], centroids.shape[0]
    dtw_kwargs = dtw_kwargs or {}

    # Compute overall centroid (use combined centroid logic)
    overall_centroid = compute_combined_centroid(X, alpha)

    # Between-cluster dispersion
    between = 0.0
    for k in range(n_clusters):
        cluster_size = np.sum(labels == k)
        dist = compute_combined_distance(centroids[k], overall_centroid,
            alpha, scaler_norm, scaler_dtw, norm_order, dtw_kwargs)
        between += cluster_size * (dist * dist)

    # Within-cluster dispersion
    within = combined_within_cluster_distance(X, labels, centroids,
        alpha, True, n_jobs, scaler_norm, scaler_dtw, norm_order, dtw_kwargs)
    within = np.sum(within)

    if within == 0:
        return np.inf

    return (between / (n_clusters - 1)) / (within / (n_samples - n_clusters))
# ==============================================================================================================
def combined_davies_bouldin_score(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    alpha: float = 0.5,
    n_jobs: int = 1,
    scaler_norm: Optional[float] = None,
    scaler_dtw: Optional[float] = None,
    norm_order: Optional[float] = None,
    dtw_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute Davies-Bouldin Score using the combined distance metric.
    """
    n_clusters = centroids.shape[0]
    dtw_kwargs = dtw_kwargs or {}

    # Compute within-cluster distances
    within = combined_within_cluster_distance(X, labels, centroids,
        alpha, False, n_jobs, scaler_norm, scaler_dtw, norm_order, dtw_kwargs)
    cluster_size = np.array([np.sum(labels == k) for k in range(n_clusters)])
    S = np.array([w / s if s != 0. else 0.0 for w, s in zip(within.tolist(), cluster_size.tolist())])

    def _calc_centroid_dist(centroids, i, j, alpha, scaler_norm, scaler_dtw, norm_order, dtw_kwargs):
        dist = compute_combined_distance(centroids[i], centroids[j],
            alpha, scaler_norm, scaler_dtw, norm_order, dtw_kwargs)
        return (i, j, dist)

    # Compute centroid-to-centroid distances
    centroid_dist_tri = Parallel(n_jobs=n_jobs)(
        delayed(_calc_centroid_dist)(centroids, i, j, alpha, scaler_norm, scaler_dtw, norm_order, dtw_kwargs)
        for i in range(n_clusters) for j in range(i+1, n_clusters)
    )

    # Fill the matrix using symmetry
    centroid_dist = np.zeros((n_clusters, n_clusters))
    for i, j, dist in centroid_dist_tri:
        centroid_dist[i, j] = dist
        centroid_dist[j, i] = dist

    # Compute Davies-Bouldin score
    DB = 0.0
    for i in range(n_clusters):
        if S[i] == 0.:
            continue

        max_ratio = -np.inf
        for j in range(n_clusters):
            if (i != j) and (S[j] != 0.):
                ratio = (S[i] + S[j]) / centroid_dist[i, j]
                max_ratio = max(max_ratio, ratio)

        if not np.isinf(max_ratio):
            DB += max_ratio

    return DB / n_clusters
# ==============================================================================================================
def combined_silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.5,
    n_jobs: int = 1,
    scaler_norm: Optional[float] = None,
    scaler_dtw: Optional[float] = None,
    norm_order: Optional[float] = None,
    dtw_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> float:

    dist_matrix = compute_distance_matrix(X, X,
        alpha, n_jobs, scaler_norm, scaler_dtw, norm_order, dtw_kwargs)

    return silhouette_score(dist_matrix, labels, metric="precomputed", **kwargs)
# ==============================================================================================================
def compute_combined_cluster_metrics(
    data, labels, centroids,
    alpha=0.5, n_jobs=1, scaler_norm=None, scaler_dtw=None, norm_order=None, dtw_kwargs=None
):
    metric = {
        'CH' : combined_calinski_harabasz_score(data, labels, centroids,
            alpha, n_jobs, scaler_norm, scaler_dtw, norm_order, dtw_kwargs),
        'DB' : combined_davies_bouldin_score(data, labels, centroids,
            alpha, n_jobs, scaler_norm, scaler_dtw, norm_order, dtw_kwargs),
        'SI' : combined_silhouette_score(data, labels,
            alpha, n_jobs, scaler_norm, scaler_dtw, norm_order, dtw_kwargs),
    }
    return metric
# ==============================================================================================================