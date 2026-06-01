import numpy as np
from typing import Tuple, List

def calculate_distance_matrix(data: np.ndarray) -> np.ndarray:
    """Calculates pairwise Euclidean distances"""
    n = data.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.sqrt(np.sum((data[i] - data[j]) ** 2))
    return dist_matrix

def k_distance(dist_matrix: np.ndarray, k: int) -> np.ndarray:
    n = dist_matrix.shape[0]
    k_dists = np.zeros(n)
    for i in range(n):
        dists = dist_matrix[i, :]
        sorted_dists = np.sort(dists)
        if k < n:
            k_dists[i] = sorted_dists[k]
        else:
            k_dists[i] = sorted_dists[-1]
    return k_dists

def kNN_SP(dist_matrix: np.ndarray, k_dists: np.ndarray) -> List[List[int]]:
    n = dist_matrix.shape[0]
    knn_subsets = []
    for i in range(n):
        # Ensure floating point issues don't skip exactly equal distances
        neighbors = np.where(dist_matrix[i] <= k_dists[i] + 1e-9)[0]
        neighbors = neighbors[neighbors != i]
        knn_subsets.append(neighbors.tolist())
    return knn_subsets

def natural_neighbor_search(dist_matrix: np.ndarray) -> Tuple[int, np.ndarray]:
    n = dist_matrix.shape[0]
    k = 1
    flag = 0
    NNAM = np.zeros((n, n), dtype=int)

    while flag == 0:
        k_dists = k_distance(dist_matrix, k)
        knns_k = kNN_SP(dist_matrix, k_dists)

        if k > 1:
            k_minus_1_dists = k_distance(dist_matrix, k - 1)
            knns_k_minus_1 = kNN_SP(dist_matrix, k_minus_1_dists)
        else:
            knns_k_minus_1 = [[] for _ in range(n)]

        for i in range(n):
            diff = list(set(knns_k[i]) - set(knns_k_minus_1[i]))
            for j in diff:
                if i in knns_k[j] and NNAM[i, j] != 1:
                    NNAM[i, j] = 1
                    NNAM[j, i] = 1

        all_have_neighbors = True
        for i in range(n):
            if np.sum(NNAM[i, :]) == 0:
                all_have_neighbors = False
                break

        if all_have_neighbors:
            flag = 1

        k += 1
        if k >= n:
            break

    lam = k - 1
    return lam, NNAM

def natural_neighbor_granularity(NNAM: np.ndarray) -> List[List[int]]:
    """Calculates NaNGS given NNAM. Formula 2-11."""
    n = NNAM.shape[0]
    NaNGS = []
    for i in range(n):
        neighbors = np.where(NNAM[i, :] == 1)[0].tolist()
        neighbors.sort()
        NaNGS.append(neighbors)
    return NaNGS

def natural_neighbor_entropy(NaNGS: List[List[int]], n: int) -> float:
    """Formula 2-14."""
    if n == 0:
        return 0.0
    entropy = 0.0
    for i in range(n):
        ratio = len(NaNGS[i]) / n
        entropy += np.log2(ratio)
    return -entropy / n

def compute_NaNRE(NaNE: float, NaNE_oi: float) -> float:
    # 2.2.15:
    # if NaNE_oi > NaNE, then 1 / (NaNE_oi / NaNE)? Wait, we verified it was 1 / (NaNE_oi / NaNE).
    # Ah, the paper text:
    # NaNREP(oi) =
    # 1                               if NaNE_oi > NaNE
    # NaNE_oi / NaNE                  if 0 <= NaNE_oi <= NaNE
    # Let's check my parsing script again!
    if NaNE_oi > NaNE:
        return 1.0
    elif NaNE == 0:
        return 1.0
    else:
        return NaNE_oi / NaNE

def compute_weight(NaNGS_i: List[int], n: int) -> float:
    return np.sqrt(len(NaNGS_i) / n)

def NaNREAD(data: np.ndarray) -> np.ndarray:
    n, m = data.shape

    # Min-Max Normalization (Data Preprocessing)
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

    def evaluate_feature_subset(subset_indices: List[int]):
        subset_data = data[:, subset_indices]
        dist_matrix = calculate_distance_matrix(subset_data)
        lam, NNAM = natural_neighbor_search(dist_matrix)
        NaNGS = natural_neighbor_granularity(NNAM)
        NaNE = natural_neighbor_entropy(NaNGS, n)

        NaNE_oi_list = []
        for i in range(n):
            keep_indices = [j for j in range(n) if j != i]
            sub_dist_matrix = dist_matrix[np.ix_(keep_indices, keep_indices)]
            sub_lam, sub_NNAM = natural_neighbor_search(sub_dist_matrix)
            sub_NaNGS = natural_neighbor_granularity(sub_NNAM)
            NaNE_oi = natural_neighbor_entropy(sub_NaNGS, n - 1)
            NaNE_oi_list.append(NaNE_oi)

        NaNRE_list = [compute_NaNRE(NaNE, NaNE_oi) for NaNE_oi in NaNE_oi_list]
        W_list = [compute_weight(NaNGS[i], n) for i in range(n)]

        return NaNRE_list, W_list

    ERMAS = np.zeros((n, m))
    WMAS = np.zeros((n, m))
    for k_idx, k in enumerate(AS_indices):
        nanre_list, w_list = evaluate_feature_subset([k])
        ERMAS[:, k_idx] = nanre_list
        WMAS[:, k_idx] = w_list

    ERMAFS = np.zeros((n, m))
    WMAFS = np.zeros((n, m))
    for k_idx, subset in enumerate(AFS):
        nanre_list, w_list = evaluate_feature_subset(subset)
        ERMAFS[:, k_idx] = nanre_list
        WMAFS[:, k_idx] = w_list

    ERMARS = np.zeros((n, m))
    WMARS = np.zeros((n, m))
    for k_idx, subset in enumerate(ARS):
        nanre_list, w_list = evaluate_feature_subset(subset)
        ERMARS[:, k_idx] = nanre_list
        WMARS[:, k_idx] = w_list

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
        scores = NaNREAD(data)
        print("NaNREAF=", scores)
    except Exception as e:
        print(f"An error occurred during algorithm execution: {e}")
