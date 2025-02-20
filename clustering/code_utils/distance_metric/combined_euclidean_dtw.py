import numpy as np

from joblib import Parallel, delayed

import tslearn.metrics as tslm
import dtaidistance as dtai


def _compute_dtw_matrix_tslearn(ts1, ts2=None, n_jobs=1, **kwargs):
    '''
    Compute DTW metric between two time series sets using tslearn.metrics.dtw.

    Parameters
    ----------
    ts1, ts2 : list of np.ndarray
        List of time series data. If ts2 is None, ts1 is used for both.
    n_jobs : int, default=1
        The number of parallel jobs to run.
    kwargs :
        Arguments passed to tslearn.metrics.dtw, for example:
        - global_constraint, sakoe_chiba_radius, itakura_max_slope, be

    Returns
    -------
    dist : np.ndarray
        Distance matrix.
    '''
    if ts2 is None:
        ts2 = ts1.copy()

    n_ts1, n_ts2 = len(ts1), len(ts2)
    # Compute all pairwise distances in parallel
    dist = Parallel(n_jobs=n_jobs)(
        delayed(tslm.dtw)(ts1[i], ts2[j], **kwargs)
        for i in range(n_ts1) for j in range(n_ts2)
    )
    # Reshape the flat list into a matrix.
    dist = np.array(dist).reshape((n_ts1, n_ts2))
    return dist
# ==============================================================================
def _compute_dtw_matrix_dtai_fast(ts1, ts2=None, **kwargs):
    '''
    Compute DTW metric between two time series sets using DTAIDistance.

    Parameters
    ----------
    ts1, ts2 : list of np.ndarray
        List of time series data. (For the dtai backend, the arrays should have dtype=np.double.)
        If ts2 is None, ts1 is used for both.
    kwargs :
        Arguments passed to dtai.dtw.distance_matrix_fast or dtai.dtw.distance.
        See https://dtaidistance.readthedocs.io/en/latest/modules/dtw.html#dtaidistance.dtw.distance_matrix_fast

    Returns
    -------
    dist : np.ndarray
        Distance matrix.
    '''
    if ts2 is None:
        dist = dtai.dtw.distance_matrix_fast(ts1, **kwargs)
    else:
        # When ts2 is provided, merge the lists and use block computation.
        n_ts1, n_ts2 = len(ts1), len(ts2)
        ts_merged = ts1 + ts2
        dist = dtai.dtw.distance_matrix_fast(
            ts_merged,
            block = ((0, n_ts1), (n_ts1, n_ts1+n_ts2)),
            compact = True, **kwargs)
        # Convert to matrix
        dist = np.array(dist).reshape((n_ts1, n_ts2))
    return dist
# ==============================================================================
def compute_dtw_matrix(ts1, ts2=None, backend_package='dtai', n_jobs=1, **kwargs):
    '''
    Compute DTW metric between two time series sets using the specified backend.

    Parameters
    ----------
    ts1, ts2 : list of np.ndarray
        List of time series data. If ts2 is None, ts1 is used for both.
        For the dtai backend, the dtype of each array should be numpy.double.
    backend_package : str, default='dtai'
        The backend package to use: either 'dtai' or 'tslearn'.
    n_jobs : int, default=1
        The number of parallel jobs to run.
    kwargs :
        Additional keyword arguments passed to the respective DTW function.
        See:
          - dtai: dtai.dtw.distance_matrix_fast or dtai.dtw.distance
          - tslearn: tslearn.metrics.dtw

    Returns
    -------
    dist : np.ndarray
        Distance matrix.
    '''
    if backend_package == 'dtai':
        return _compute_dtw_matrix_dtai_fast(ts1, ts2, **kwargs)
    elif backend_package == 'tslearn':
        return _compute_dtw_matrix_tslearn(ts1, ts2, n_jobs=n_jobs, **kwargs)
    else:
        raise ValueError(f'Invalid backend package: {backend_package}')
# =================================================================================