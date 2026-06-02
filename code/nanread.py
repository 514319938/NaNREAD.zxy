import numpy as np
from typing import Tuple, List
from scipy.spatial.distance import pdist, squareform
import multiprocessing
from functools import partial

# ========== 从上方正确代码复制【精确NNS函数：支持等值tie、数学严格】 ==========
def calculate_distance_matrix(X, subset_indices=None, q=2):
    if subset_indices is not None:
        X_sub = X[:, subset_indices]
    else:
        X_sub = X
    if X_sub.ndim == 1:
        X_sub = X_sub.reshape(-1, 1)
    if q==2:
        dist_array=pdist(X_sub, metric="euclidean")
    else:
        dist_array = pdist(X_sub, metric='minkowski', p=q)
    dist_mat = squareform(dist_array)
    return dist_mat

def natural_neighbor_search(dist_mat):
    n = dist_mat.shape[0]
    dist_mat_inf = dist_mat.copy()
    np.fill_diagonal(dist_mat_inf, np.inf)
    sorted_idx = np.argsort(dist_mat_inf, axis=1, kind='stable')
    kNNS_matrix = np.zeros((n, n), dtype=bool)
    k = 1
    while True:
        if k - 1 < n:
            kth_dists = dist_mat_inf[np.arange(n), sorted_idx[:, k - 1]]
            new_neighbors = dist_mat_inf <= kth_dists[:, np.newaxis]
            kNNS_matrix = kNNS_matrix | new_neighbors
        NNAM = (kNNS_matrix & kNNS_matrix.T).astype(int)
        if np.all(np.sum(NNAM, axis=1) != 0):
            break
        k += 1
        if k > n:
            break
    lambda_P = k
    return lambda_P, NNAM

def calculate_nane(nnam):
    n = nnam.shape[0]
    sizes = np.sum(nnam, axis=1)
    sizes = np.maximum(sizes, 1)
    probabilities = sizes / n
    nane = -1.0 / n * np.sum(np.log2(probabilities))
    return nane, sizes

def calculate_nane_fast(dist_mat):
    lambda_P, nnam_P = natural_neighbor_search(dist_mat)
    nane_val, sizes = calculate_nane(nnam_P)
    return nane_val, sizes

# ========== 原有辅助函数小幅兼容适配，入口不动 ==========
def natural_neighbor_granularity(NNAM: np.ndarray) -> List[List[int]]:
    n = NNAM.shape[0]
    NaNGS = []
    for i in range(n):
        neighbors = np.where(NNAM[i, :] == 1)[0].tolist()
        neighbors.sort()
        NaNGS.append(neighbors)
    return NaNGS

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
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    dist_mat_sub = dist_matrix[mask][:, mask]
    sub_NaNE, _ = calculate_nane_fast(dist_mat_sub)
    return sub_NaNE

def evaluate_feature_subset_parallel(subset_indices, data, pool):
    n = data.shape[0]
    dist_matrix = calculate_distance_matrix(data, subset_indices)
    NaNE, sizes = calculate_nane_fast(dist_matrix)
    _, NNAM = natural_neighbor_search(dist_matrix)
    NaNGS = natural_neighbor_granularity(NNAM)

    func = partial(compute_NaNE_oi, dist_matrix=dist_matrix, n=n)
    if pool is not None:
        NaNE_oi_list = pool.map(func, range(n))
    else:
        NaNE_oi_list = [func(i) for i in range(n)]

    NaNRE_list = [compute_NaNRE(NaNE, oi) for oi in NaNE_oi_list]
    W_list = [compute_weight(NaNGS[i], n) for i in range(n)]
    return NaNRE_list, W_list

# ========== 对外入口：NaNREAD(data, n_jobs=-1)【和上层调用完全兼容，不用改实验代码】 ==========
def NaNREAD(data: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    n, m = data.shape
    # min-max归一化（和上段代码统一）
    data_std = np.zeros_like(data, dtype=float)
    for j in range(m):
        col_min = np.min(data[:, j])
        col_max = np.max(data[:, j])
        diff = col_max - col_min
        if diff < 1e-12:
            data_std[:, j] = 0.0
        else:
            data_std[:, j] = (data[:, j] - col_min) / diff
    X = data_std

    # 单属性NaNE排序生成AS
    nane_single = []
    for k in range(m):
        dist_mat = calculate_distance_matrix(X, [k])
        nane, _ = calculate_nane_fast(dist_mat)
        nane_single.append((nane, k))
    nane_single.sort()
    AS_indices = [idx for val, idx in nane_single]

    # 生成AFS、ARS
    AFS = []
    cur = []
    for idx in AS_indices:
        cur.append(idx)
        AFS.append(cur.copy())
    ARS = []
    cur = []
    for idx in reversed(AS_indices):
        cur.append(idx)
        ARS.append(cur.copy())

    ERMAS = np.zeros((n, m))
    WMAS = np.zeros((n, m))
    ERMAFS = np.zeros((n, m))
    WMAFS = np.zeros((n, m))
    ERMARS = np.zeros((n, m))
    WMARS = np.zeros((n, m))

    # 进程池
    if n_jobs == -1:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
    elif n_jobs > 1:
        pool = multiprocessing.Pool(n_jobs)
    else:
        pool = None

    try:
        for k_idx, k in enumerate(AS_indices):
            nanre_list, w_list = evaluate_feature_subset_parallel([k], X, pool)
            ERMAS[:, k_idx] = nanre_list
            WMAS[:, k_idx] = w_list
        for k_idx, subset in enumerate(AFS):
            nanre_list, w_list = evaluate_feature_subset_parallel(subset, X, pool)
            ERMAFS[:, k_idx] = nanre_list
            WMAFS[:, k_idx] = w_list
        for k_idx, subset in enumerate(ARS):
            nanre_list, w_list = evaluate_feature_subset_parallel(subset, X, pool)
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
        sum_val = np.sum(AERM[i, :] * AWM[i, :])
        NaNREAF[i] = 1.0 - sum_val / m
    return NaNREAF

# 自测示例
if __name__ == "__main__":
    import scipy.io
    filepath = 'code/Example.mat'
    mat = scipy.io.loadmat(filepath)
    data = None
    for k,v in mat.items():
        if not k.startswith('__') and isinstance(v,np.ndarray):
            data = v
            break
    score = NaNREAD(data,n_jobs=1)
    print(score)
