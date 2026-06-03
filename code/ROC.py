import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ====== 配置区 ======
EXPERIMENTAL_RESULTS_DIR = r"D:\Experimental_results"
DATA_DIR = "data"
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "_figures", "ROC")

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

# 映射算法到对应的文件夹名称，特别是 VAE -> AutoEncoder_results
algo_dir_map = {
    'VAE': 'AutoEncoder_results'
}
for algo in algorithms:
    if algo not in algo_dir_map and algo != 'NaNREAD':
        algo_dir_map[algo] = f"{algo}_results"

colors = plt.cm.tab20(np.linspace(0, 1, len(algorithms)))

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
    """递归搜索基准目录下包含数据集名称的无参数 .mat 文件"""
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

    # 优先找不带参数的 (没有 lam 也没有 param)
    for f in matched_files:
        if 'lam' not in f.lower() and 'param' not in f.lower():
            return f

    # 如果全带有参数，则取最短的文件名，通常是最基础的文件
    matched_files.sort(key=len)
    return matched_files[0]

def get_outlier_scores(dataset, algo):
    # ==================== NaNREAD ====================
    if algo == 'NaNREAD':
        path1 = os.path.join(RESULTS_DIR, dataset, f"{dataset}.mat")
        path2 = os.path.join(RESULTS_DIR, f"{dataset}.mat")

        if os.path.exists(path1):
            mat_path = path1
        elif os.path.exists(path2):
            mat_path = path2
        else:
            print(f"[NaNREAD] 未找到 {dataset} 结果文件")
            return None

        try:
            mat = scipy.io.loadmat(mat_path)
            # 根据之前的记录，NaNREAD 使用的是 opt_out_scores
            if 'opt_out_scores' in mat:
                scores_array = mat['opt_out_scores']
            elif dataset in mat:
                scores_array = mat[dataset]
            else:
                print(f"❌[NaNREAD]无 'opt_out_scores' 也无 '{dataset}' 变量，键：{list(mat.keys())}")
                return None

            if scores_array.ndim >= 2 and scores_array.shape[1] >= 1:
                scores = scores_array[:, 0].flatten()
            else:
                scores = scores_array.flatten()
            print(f"✅[NaNREAD] {dataset}读取成功,分数长度:{len(scores)}")
            return scores
        except Exception as e:
            print(f"[NaNREAD]读取异常:{e}")
            return None

    # ==================== 其余对比算法 ====================
    else:
        algo_dir = algo_dir_map[algo]
        algo_base = os.path.join(EXPERIMENTAL_RESULTS_DIR, algo_dir)

        # 使用强化搜索，递归查找包含 dataset 的对应 mat
        mat_path = search_mat_file(algo_base, dataset)

        if not mat_path:
            print(f"未找到 {algo} 在 {dataset} 上的结果 (.mat 文件)")
            return None

        try:
            mat = scipy.io.loadmat(mat_path)
            # 对比算法大多数也是 opt_out_scores
            if 'opt_out_scores' in mat:
                scores_array = mat['opt_out_scores']
            else:
                # 尝试其他可能的键名
                possible_keys = [k for k in mat.keys() if not k.startswith('__')]
                if len(possible_keys) > 0:
                    scores_array = mat[possible_keys[0]]
                else:
                    print(f"警告: {mat_path} 中找不到有效的分数变量")
                    return None

            if scores_array.ndim >= 2 and scores_array.shape[1] >= 1:
                scores = scores_array[:, 0].flatten()
            else:
                scores = scores_array.flatten()
            print(f"✅[{algo}] {dataset}读取成功,分数长度:{len(scores)}")
            return scores
        except Exception as e:
            print(f"读取失败 {mat_path}: {e}")
            return None

def plot_roc_for_dataset(dataset):
    print(f"\n正在处理: {dataset}")
    y_true = get_true_labels(dataset)
    if y_true is None:
        return

    plt.figure(figsize=(10, 8))

    for idx, algo in enumerate(algorithms):
        scores = get_outlier_scores(dataset, algo)
        if scores is None:
            continue

        # 强制对齐长度 (如用户所提供的)
        if len(scores) > len(y_true):
            scores = scores[:len(y_true)]

        if len(y_true) != len(scores):
            print(f"⚠️ {algo} 长度不匹配，跳过 (y_true:{len(y_true)}, scores:{len(scores)})")
            continue

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[idx], lw=2, label=f"{algo} (AUC={roc_auc:.4f})")

    plt.plot([0,1],[0,1], 'k--', lw=1.5)
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('FPR', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.title(f'ROC Curve on {dataset}', fontsize=16)

    handles, _ = plt.gca().get_legend_handles_labels()
    if len(handles) > 0:
        plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f"{dataset}_ROC.pdf")
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"✅ 已生成 ROC 图: {save_path}")

def main():
    if not os.path.exists(EXPERIMENTAL_RESULTS_DIR):
        print(f"==== 提示 ====")
        print(f"当前配置的外部结果目录不存在: {EXPERIMENTAL_RESULTS_DIR}")
        print(f"本脚本将在找不到对比算法数据时跳过绘制它们的曲线。")
        print(f"请在本地运行前，确保 EXPERIMENTAL_RESULTS_DIR 变量指向正确的 'Experimental_results' 文件夹路径！\n")

    for dataset in datasets:
        plot_roc_for_dataset(dataset)

    print("\n🎉 全部 ROC 图像生成完毕。")

if __name__ == '__main__':
    main()
