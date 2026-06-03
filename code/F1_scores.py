import os
import numpy as np
import scipy.io
import pandas as pd
from sklearn.metrics import precision_recall_curve

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ====== 配置区 ======
EXPERIMENTAL_RESULTS_DIR = r"D:\Experimental_results"
DATA_DIR = "data"
RESULTS_DIR = "results"

ensure_dir(RESULTS_DIR)

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

algo_dir_map = {
    'VAE': 'AutoEncoder_results'
}
for algo in algorithms:
    if algo not in algo_dir_map and algo != 'NaNREAD':
        algo_dir_map[algo] = f"{algo}_results"

def get_true_labels(dataset):
    data_path = os.path.join(DATA_DIR, f"{dataset}.mat")
    if not os.path.exists(data_path):
        print(f"警告: 找不到数据集文件 {data_path}")
        return None

    mat = scipy.io.loadmat(data_path)
    data = None
    for key, val in mat.items():
        if not key.startswith('__') and hasattr(val, 'shape') and len(val.shape) > 1:
            data = np.array(val)
            break

    if data is None:
        print(f"警告: 数据集 {dataset} 中未找到有效数据矩阵")
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

def compute_best_f1(y_true, scores):
    """计算所有可能阈值下的最大 F1 分数"""
    # 处理 nan/inf 异常值，防止报错
    scores = np.nan_to_num(scores, nan=0.0, posinf=np.nanmax(scores[scores != np.inf]), neginf=0.0)

    # 确保 y_true 是 0/1 (假定通常 1 是异常，如果有非0/1，简单判断非 0 即可)
    y_true_binary = (y_true != 0).astype(int)

    # precision_recall_curve 提供不同阈值下的精确率和召回率
    precision, recall, _ = precision_recall_curve(y_true_binary, scores)

    # 防止分母为0
    numerator = 2 * recall * precision
    denominator = recall + precision

    # 遇到 0/0 返回 0
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)

    return np.max(f1_scores)

def main():
    print(f"正在生成最佳 F1 分数表格...")

    # 构建一个空的 DataFrame，行是算法，列是数据集
    f1_df = pd.DataFrame(index=algorithms, columns=datasets)

    for dataset in datasets:
        print(f"处理数据集: {dataset}")
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

            best_f1 = compute_best_f1(y_true, scores)
            f1_df.loc[algo, dataset] = best_f1

    # 保存为 Excel 文件
    output_path = os.path.join(RESULTS_DIR, "F1_scores.xlsx")
    f1_df.to_excel(output_path, index=True, float_format="%.4f")
    print(f"\n🎉 最佳 F1 分数计算完成，已保存至: {output_path}")

if __name__ == '__main__':
    main()
