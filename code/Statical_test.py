import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.metrics import roc_auc_score
import math

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 配置区
EXPERIMENTAL_RESULTS_DIR = r"D:\Experimental_results"
DATA_DIR = "data"
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "_figures")

ensure_dir(FIGURES_DIR)

datasets = [
    'breast_cancer_variant1', 'chess_nowin_34_variant1',
    'monks_0_4_variant1', 'diabetes_tested_positive_26_variant1',
    'tic_tac_toe_negative_12_variant1', 'tic_tac_toe_negative_26_variant1',
    'zoo_variant1', 'cardiotocography_2and3_33_variant1', 'glass',
    'ionosphere_b_24_variant1', 'letter', 'pima_TRUE_55_variant1',
    'vowels', 'annealing_variant1', 'bands_band_6_variant1'
]

algorithms = [
    'SEQ', 'IE', 'ITB', 'WDOD', 'ODGrCR', 'ApproE', 'VarE', 'ILGNI',
    'MFIOD', 'VAE', 'NaNREAD'
]

# 映射算法到对应的文件夹名称
algo_dir_map = {
    'VAE': 'AutoEncoder_results'
}
for algo in algorithms:
    if algo not in algo_dir_map and algo != 'NaNREAD':
        algo_dir_map[algo] = f"{algo}_results"

# ==================== CD Diagram 生成逻辑 ====================

def compute_CD(avranks, n, alpha="0.05", test="nemenyi"):
    """
    Returns critical difference for Nemenyi or Bonferroni-Dunn test
    """
    k = len(avranks)
    if test == "nemenyi":
        # Values from table 5(b) in Demšar (2006)
        # alpha = 0.05
        q_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
                8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
                13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458,
                18: 3.489, 19: 3.517, 20: 3.544}
        # alpha = 0.1
        q_10 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693,
                8: 2.780, 9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030,
                13: 3.077, 14: 3.120, 15: 3.159, 16: 3.196, 17: 3.230,
                18: 3.261, 19: 3.291, 20: 3.319}
        # alpha = 0.01
        q_01 = {2: 2.576, 3: 2.913, 4: 3.113, 5: 3.255, 6: 3.364, 7: 3.452,
                8: 3.526, 9: 3.590, 10: 3.646, 11: 3.696, 12: 3.741,
                13: 3.781, 14: 3.818, 15: 3.853, 16: 3.884, 17: 3.914,
                18: 3.941, 19: 3.967, 20: 3.992}

        q = {"0.05": q_05, "0.1": q_10, "0.01": q_01}.get(alpha)
        if q is None:
            raise ValueError(f"Invalid alpha value. Supported: 0.05, 0.1, 0.01")
        if k not in q:
            raise ValueError(f"Number of algorithms {k} is outside the supported range (2-20)")

        return q[k] * np.sqrt((k * (k + 1)) / (6.0 * n))

def graph_ranks(avranks, names, p_values=None, cd=None, cdmethod=None, lowv=None, highv=None, width=6, textspace=1, reverse=False, filename=None, **kwargs):
    """
    Draws a CD graph, which is used to display the differences in methods' performance.
    """
    width = float(width)
    textspace = float(textspace)

    def round_to_1(x):
        if x == 0:
            return 0
        return round(x, -int(math.floor(math.log10(abs(x)))))

    k = len(avranks)

    if lowv is None:
        lowv = min(1, int(math.floor(min(avranks))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(avranks))))

    cline = 0.4
    khalf = int(math.ceil(k / 2.0))
    distanceh = 0.25

    fig = plt.figure(figsize=(width, khalf * distanceh + 1.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    scaley = 1.0 - 0.2 / (khalf * distanceh + 1.5)
    d = (highv - lowv)
    bl = 0.1
    br = 0.9
    rankline = br - bl

    ax.plot([bl, br], [scaley, scaley], color="k")

    for a in range(lowv, highv + 1):
        tick_x = bl + rankline * (a - lowv) / d
        ax.plot([tick_x, tick_x], [scaley - 0.05, scaley + 0.05], color="k")
        ax.text(tick_x, scaley + 0.1, str(a), ha="center", va="bottom", fontsize=12)

    if cd is not None:
        cdy = scaley + 0.2
        cdx = bl + rankline * cd / d
        ax.plot([bl, cdx], [cdy, cdy], color="k", lw=2)
        ax.plot([bl, bl], [cdy - 0.05, cdy + 0.05], color="k", lw=2)
        ax.plot([cdx, cdx], [cdy - 0.05, cdy + 0.05], color="k", lw=2)
        ax.text((bl + cdx) / 2.0, cdy + 0.05, "CD", ha="center", va="bottom", fontsize=12)

    sorted_ranks, sorted_names = zip(*sorted(zip(avranks, names)))

    if reverse:
        sorted_ranks = list(reversed(sorted_ranks))
        sorted_names = list(reversed(sorted_names))

    if cd is not None:
        def get_lines(ranks):
            lines = []
            for i in range(len(ranks)):
                for j in range(len(ranks) - 1, i, -1):
                    if ranks[j] - ranks[i] <= cd:
                        contained = False
                        for l in lines:
                            if l[0] <= i and l[1] >= j:
                                contained = True
                                break
                        if not contained:
                            lines.append((i, j))
            return lines
        lines = get_lines(sorted_ranks)
    else:
        lines = []

    for i in range(math.ceil(k / 2.0)):
        rank = sorted_ranks[i]
        name = sorted_names[i]
        rank_x = bl + rankline * (rank - lowv) / d
        text_y = scaley - cline - distanceh * i
        ax.plot([rank_x, rank_x], [scaley, text_y], color="k")
        ax.plot([rank_x, bl - 0.05], [text_y, text_y], color="k")
        ax.text(bl - 0.06, text_y, f"{name}", ha="right", va="center", fontsize=12)

    for i in range(math.ceil(k / 2.0), k):
        rank = sorted_ranks[i]
        name = sorted_names[i]
        rank_x = bl + rankline * (rank - lowv) / d
        text_y = scaley - cline - distanceh * (k - 1 - i)
        ax.plot([rank_x, rank_x], [scaley, text_y], color="k")
        ax.plot([rank_x, br + 0.05], [text_y, text_y], color="k")
        ax.text(br + 0.06, text_y, f"{name}", ha="left", va="center", fontsize=12)

    if lines:
        line_height = scaley - 0.15
        line_spacing = 0.05
        for idx, line in enumerate(lines):
            start_x = bl + rankline * (sorted_ranks[line[0]] - lowv) / d
            end_x = bl + rankline * (sorted_ranks[line[1]] - lowv) / d
            y = line_height - idx * line_spacing
            ax.plot([start_x, end_x], [y, y], color="r", lw=3)

    if filename:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close()

# ==================== 数据读取与处理 ====================

def get_true_labels(dataset):
    data_path = os.path.join(DATA_DIR, f"{dataset}.mat")
    if not os.path.exists(data_path):
        return None

    mat = scipy.io.loadmat(data_path)
    data = None
    for key, val in mat.items():
        if not key.startswith('__') and hasattr(val, 'shape') and len(val.shape) > 1:
            data = np.array(val)
            break

    if data is None:
        return None

    y = data[:, -1]
    return y

def search_mat_file(base_dir, dataset):
    matched_files = []
    if not os.path.exists(base_dir):
        return None

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.mat') and dataset in file:
                full_path = os.path.join(root, file)
                matched_files.append(full_path)

    if not matched_files:
        return None

    for f in matched_files:
        if 'lam' not in f.lower() and 'param' not in f.lower():
            return f

    matched_files.sort(key=len)
    return matched_files[0]

def get_outlier_scores(dataset, algo):
    if algo == 'NaNREAD':
        path1 = os.path.join(RESULTS_DIR, dataset, f"{dataset}.mat")
        path2 = os.path.join(RESULTS_DIR, f"{dataset}.mat")

        if os.path.exists(path1):
            mat_path = path1
        elif os.path.exists(path2):
            mat_path = path2
        else:
            return None

        try:
            mat = scipy.io.loadmat(mat_path)
            if 'opt_out_scores' in mat:
                scores_array = mat['opt_out_scores']
            elif dataset in mat:
                scores_array = mat[dataset]
            else:
                return None

            if scores_array.ndim >= 2 and scores_array.shape[1] >= 1:
                scores = scores_array[:, 0].flatten()
            else:
                scores = scores_array.flatten()
            return scores
        except Exception:
            return None
    else:
        algo_dir = algo_dir_map[algo]
        algo_base = os.path.join(EXPERIMENTAL_RESULTS_DIR, algo_dir)
        mat_path = search_mat_file(algo_base, dataset)

        if not mat_path:
            return None

        try:
            mat = scipy.io.loadmat(mat_path)
            if 'opt_out_scores' in mat:
                scores_array = mat['opt_out_scores']
            else:
                possible_keys = [k for k in mat.keys() if not k.startswith('__')]
                if len(possible_keys) > 0:
                    scores_array = mat[possible_keys[0]]
                else:
                    return None

            if scores_array.ndim >= 2 and scores_array.shape[1] >= 1:
                scores = scores_array[:, 0].flatten()
            else:
                scores = scores_array.flatten()
            return scores
        except Exception:
            return None


def main():
    print("Computing AUC scores for Nemenyi tests...")

    auc_df = pd.DataFrame(index=datasets, columns=algorithms)

    for dataset in datasets:
        y_true = get_true_labels(dataset)
        if y_true is None:
            continue

        for algo in algorithms:
            scores = get_outlier_scores(dataset, algo)
            if scores is None:
                continue

            # 强制对齐长度
            if len(scores) > len(y_true):
                scores = scores[:len(y_true)]
            if len(y_true) != len(scores):
                continue

            # 避免全 0 或全是常数时计算报错，给一些保护
            if len(np.unique(y_true)) > 1:
                try:
                    auc_val = roc_auc_score((y_true != 0).astype(int), scores)
                    auc_df.loc[dataset, algo] = auc_val
                except:
                    pass

    # 过滤空数据，用 0 填充，表示该算法未能预测出该数据集
    auc_df = auc_df.fillna(0.0)

    auc_matrix = auc_df.values
    if len(auc_matrix) == 0:
        print("No valid datasets with scores for selected algorithms.")
        return

    # 计算 Rank (AUC越大越好，因此我们给负值排名)
    from scipy.stats import rankdata
    ranks = np.array([rankdata(-row) for row in auc_matrix])
    average_ranks = np.mean(ranks, axis=0)

    num_datasets = len(auc_matrix)
    print(f"Computing Nemenyi tests over {num_datasets} datasets for {len(algorithms)} algorithms.")
    print("Average ranks:", dict(zip(algorithms, average_ranks)))

    alphas = ["0.05", "0.1", "0.01"]

    for alpha in alphas:
        # Nemenyi CD distance
        cd = compute_CD(average_ranks, num_datasets, alpha=alpha)

        # 保存图片
        alpha_str = str(alpha).replace('.', '')
        out_filename = os.path.join(FIGURES_DIR, f"Nemenyi_CD_Diagram_alpha_{alpha_str}.pdf")

        graph_ranks(average_ranks, algorithms, cd=cd, width=10, textspace=1.5, filename=out_filename)

        print(f"✅ Generated Nemenyi CD diagram for alpha {alpha}: {out_filename}")

if __name__ == '__main__':
    main()
