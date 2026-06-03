import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ====== 配置区 ======
# 请根据您本地的实际路径修改 EXPERIMENTAL_RESULTS_DIR
# 例如: EXPERIMENTAL_RESULTS_DIR = r"D:\Experimental_results"
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

# 映射算法到对应的文件夹名称，特别是 VAE -> AutoEncoder_results
algo_dir_map = {
    'VAE': 'AutoEncoder_results'
}
for algo in algorithms:
    if algo not in algo_dir_map and algo != 'NaNREAD':
        algo_dir_map[algo] = f"{algo}_results"

# 为画图提供一些不同样式
colors = plt.cm.tab20(np.linspace(0, 1, len(algorithms)))

def get_true_labels(dataset):
    """从原始数据读取真实标签 y"""
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
        print(f"警告: 数据集 {dataset} 中未找到有效的数据矩阵")
        return None

    y = data[:, -1]
    return y

def get_outlier_scores(dataset, algo):
    """获取指定算法在该数据集上的异常得分"""
    if algo == 'NaNREAD':
        # NaNREAD 结果存在 results/ 目录下
        mat_path = os.path.join(RESULTS_DIR, dataset, f"{dataset}.mat")
    else:
        # 其他对比算法存在 Experimental_results 目录下
        algo_dir = algo_dir_map[algo]
        mat_path = os.path.join(EXPERIMENTAL_RESULTS_DIR, algo_dir, dataset, f"{dataset}_{algo}.mat")

    if not os.path.exists(mat_path):
        print(f"未找到算法 {algo} 在 {dataset} 上的结果文件: {mat_path}")
        return None

    try:
        mat = scipy.io.loadmat(mat_path)
        if 'opt_out_scores' in mat:
            scores = mat['opt_out_scores'].flatten()
            return scores
        else:
            print(f"警告: 文件 {mat_path} 中不包含 'opt_out_scores' 变量")
            return None
    except Exception as e:
        print(f"读取文件 {mat_path} 出错: {e}")
        return None

def plot_roc_for_dataset(dataset):
    print(f"正在处理数据集: {dataset} ...")
    y_true = get_true_labels(dataset)
    if y_true is None:
        return

    plt.figure(figsize=(10, 8))

    for idx, algo in enumerate(algorithms):
        scores = get_outlier_scores(dataset, algo)
        if scores is None:
            continue

        if len(y_true) != len(scores):
            print(f"警告: 数据集 {dataset} 和 算法 {algo} 的样本数量不一致 (y_true:{len(y_true)}, scores:{len(scores)})")
            continue

        # 计算 ROC 曲线
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

        # 绘制曲线
        plt.plot(fpr, tpr, color=colors[idx], lw=2, label=f"{algo} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curve on {dataset}', fontsize=16)
    # Only show legend if we actually plotted anything
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图片
    fig_path = os.path.join(FIGURES_DIR, f"{dataset}_ROC.pdf")
    plt.savefig(fig_path, format='pdf')
    plt.close()
    print(f"已生成 ROC 图: {fig_path}")

def main():
    if not os.path.exists(EXPERIMENTAL_RESULTS_DIR):
        print(f"==== 提示 ====")
        print(f"当前配置的外部结果目录不存在: {EXPERIMENTAL_RESULTS_DIR}")
        print(f"本脚本将在找不到对比算法数据时跳过绘制它们的曲线。")
        print(f"请在本地运行前，确保 EXPERIMENTAL_RESULTS_DIR 变量指向正确的 'Experimental_results' 文件夹路径！\n")

    for dataset in datasets:
        plot_roc_for_dataset(dataset)

    print("所有 ROC 图像生成完毕。")

if __name__ == '__main__':
    main()
