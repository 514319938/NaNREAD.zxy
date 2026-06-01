import numpy as np
from typing import Tuple, List
from scipy.spatial.distance import pdist, squareform
import multiprocessing
from functools import partial

def calculate_distance_matrix(data: np.ndarray) -> np.ndarray:
    return squareform(pdist(data, metric='euclidean'))

def natural_neighbor_search(dist_matrix: np.ndarray) -> Tuple[int, np.ndarray]:
    n = dist_matrix.shape[0]
    k = 1
    NNAM = np.zeros((n, n), dtype=int)

    sorted_dist_matrix = np.sort(dist_matrix, axis=1)

    def get_k_dists(k_val):
        idx = min(k_val, n - 1)
        return sorted_dist_matrix[:, idx]

    knns_k_minus_1_mask = np.zeros((n, n), dtype=bool)
    unconnected_count = n
    has_neighbor = np.zeros(n, dtype=bool)

    while True:
        k_dists = get_k_dists(k)
        knns_k_mask = dist_matrix <= (k_dists[:, None] + 1e-9)
        np.fill_diagonal(knns_k_mask, False)

        diff_mask = knns_k_mask & ~knns_k_minus_1_mask
        new_edges = diff_mask & knns_k_mask.T
        sym_new_edges = new_edges | new_edges.T

        NNAM[sym_new_edges] = 1

        if unconnected_count > 0:
            newly_connected = sym_new_edges.any(axis=1)
            just_connected = newly_connected & ~has_neighbor
            has_neighbor |= just_connected
            unconnected_count -= just_connected.sum()

        if unconnected_count == 0:
            break

        knns_k_minus_1_mask = knns_k_mask
        k += 1
        if k >= n:
            break

    lam = k - 1 if k > 1 else 1
    return lam, NNAM

def natural_neighbor_granularity(NNAM: np.ndarray) -> List[List[int]]:
    n = NNAM.shape[0]
    NaNGS = []
    for i in range(n):
        neighbors = np.where(NNAM[i, :] == 1)[0].tolist()
        neighbors.sort()
        NaNGS.append(neighbors)
    return NaNGS

def natural_neighbor_entropy(NaNGS: List[List[int]], n: int) -> float:
    if n == 0:
        return 0.0
    entropy = 0.0
    for i in range(n):
        ratio = len(NaNGS[i]) / n
        if ratio > 0:
            entropy += np.log2(ratio)
    return -entropy / n

def compute_NaNRE(NaNE: float, NaNE_oi: float) -> float:
    if NaNE_oi > NaNE:
        return 1.0
    elif NaNE == 0:
        return 1.0
    else:
        return NaNE_oi / NaNE

def compute_weight(NaNGS_i: List[int], n: int) -> float:
    return np.sqrt(len(NaNGS_i) / n)

def compute_NaNE_oi(i, dist_matrix, n):
    keep_indices = [j for j in range(n) if j != i]
    sub_dist_matrix = dist_matrix[np.ix_(keep_indices, keep_indices)]
    sub_lam, sub_NNAM = natural_neighbor_search(sub_dist_matrix)
    sub_NaNGS = natural_neighbor_granularity(sub_NNAM)
    return natural_neighbor_entropy(sub_NaNGS, n - 1)

def evaluate_feature_subset_parallel(subset_indices, data, pool):
    n = data.shape[0]
    subset_data = data[:, subset_indices]
    dist_matrix = calculate_distance_matrix(subset_data)
    lam, NNAM = natural_neighbor_search(dist_matrix)
    NaNGS = natural_neighbor_granularity(NNAM)
    NaNE = natural_neighbor_entropy(NaNGS, n)

    func = partial(compute_NaNE_oi, dist_matrix=dist_matrix, n=n)

    if pool is not None:
        NaNE_oi_list = pool.map(func, range(n))
    else:
        NaNE_oi_list = [func(i) for i in range(n)]

    NaNRE_list = [compute_NaNRE(NaNE, NaNE_oi) for NaNE_oi in NaNE_oi_list]
    W_list = [compute_weight(NaNGS[i], n) for i in range(n)]

    return NaNRE_list, W_list

def NaNREAD(data: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    n, m = data.shape

    data_std = np.zeros_like(data, dtype=float)
    for j in range(m):
        col_min = np.min(data[:, j])
        col_max = np.max(data[:, j])
        if col_max > col_min:
            data_std[:, j] = (data[:, j] - col_min) / (col_max - col_min)
        else:
            data_std[:, j] = 0.0
    data = data_std

    nane_single = []
    for k in range(m):
        feature_data = data[:, k:k+1]
        dist_matrix = calculate_distance_matrix(feature_data)
        lam, NNAM = natural_neighbor_search(dist_matrix)
        NaNGS = natural_neighbor_granularity(NNAM)
        NaNE = natural_neighbor_entropy(NaNGS, n)
        nane_single.append((NaNE, k))

    nane_single.sort()
    AS_indices = [idx for val, idx in nane_single]

    AFS = []
    current_set = []
    for k in AS_indices:
        current_set.append(k)
        AFS.append(list(current_set))

    ARS = []
    current_set = []
    for k in reversed(AS_indices):
        current_set.append(k)
        ARS.append(list(current_set))

    ERMAS = np.zeros((n, m))
    WMAS = np.zeros((n, m))
    ERMAFS = np.zeros((n, m))
    WMAFS = np.zeros((n, m))
    ERMARS = np.zeros((n, m))
    WMARS = np.zeros((n, m))

    if n_jobs == -1:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
    elif n_jobs > 1:
        pool = multiprocessing.Pool(n_jobs)
    else:
        pool = None

    try:
        for k_idx, k in enumerate(AS_indices):
            nanre_list, w_list = evaluate_feature_subset_parallel([k], data, pool)
            ERMAS[:, k_idx] = nanre_list
            WMAS[:, k_idx] = w_list

        for k_idx, subset in enumerate(AFS):
            nanre_list, w_list = evaluate_feature_subset_parallel(subset, data, pool)
            ERMAFS[:, k_idx] = nanre_list
            WMAFS[:, k_idx] = w_list

        for k_idx, subset in enumerate(ARS):
            nanre_list, w_list = evaluate_feature_subset_parallel(subset, data, pool)
            ERMARS[:, k_idx] = nanre_list
            WMARS[:, k_idx] = w_list
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    AERM = (ERMAS + ERMAFS + ERMARS) / 3.0
    AWM = (WMAS + WMAFS + WMARS) / 3.0

    NaNREAF = np.zeros(n)
    for i in range(n):
        sum_val = 0.0
        for k in range(m):
            sum_val += AERM[i, k] * AWM[i, k]
        NaNREAF[i] = 1.0 - sum_val / m

    return NaNREAF

if __name__ == "__main__":
    import scipy.io
    import sys

    def load_mat(filepath):
        try:
            return scipy.io.loadmat(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    filepath = 'code/Example.mat'
    data_dict = load_mat(filepath)
    if data_dict is None:
        sys.exit(1)

    data = None
    for key, val in data_dict.items():
        if not key.startswith('__') and hasattr(val, 'shape'):
            data = np.array(val)
            break

    if data is None:
        print("Could not find a valid variable in the file.")
        sys.exit(1)

    try:
        scores = NaNREAD(data, n_jobs=1)
        print("NaNREAF=", scores)
    except Exception as e:
        print(f"An error occurred during algorithm execution: {e}")
