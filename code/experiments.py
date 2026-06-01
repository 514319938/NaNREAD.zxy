import os
import time
import numpy as np
import scipy.io
import xlwt
from sklearn.metrics import roc_auc_score
from nanread import NaNREAD
import multiprocessing

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_mat(filepath):
    try:
        return scipy.io.loadmat(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def process_dataset(filepath):
    dataset_name = os.path.basename(filepath).replace('.mat', '')
    if dataset_name == 'Example':
        return

    result_dir = os.path.join('results', dataset_name)
    xls_out_path = os.path.join(result_dir, f"{dataset_name}.xls")
    if os.path.exists(xls_out_path):
        print(f"Already processed {dataset_name}. Skipping...", flush=True)
        return

    print(f"Processing {dataset_name}...", flush=True)
    data_dict = load_mat(filepath)
    if data_dict is None:
        return

    data = None
    for key, val in data_dict.items():
        if not key.startswith('__') and hasattr(val, 'shape') and len(val.shape) > 1:
            data = np.array(val)
            break

    if data is None:
        print(f"Skipping {dataset_name}: No valid array found.", flush=True)
        return

    n = data.shape[0]
    if n >= 2000:
        print(f"Skipping {dataset_name}: n={n} >= 2000", flush=True)
        return

    X = data[:, :-1]
    y = data[:, -1]

    start_time = time.time()
    try:
        # Avoid nested multiprocessing: process datasets sequentially but parallelize NaNREAD
        opt_out_scores = NaNREAD(X, n_jobs=-1)
    except Exception as e:
        print(f"Error running NaNREAD on {dataset_name}: {e}", flush=True)
        return
    opt_time = time.time() - start_time

    try:
        opt_ROC_AUC = roc_auc_score(y, opt_out_scores)
    except Exception as e:
        print(f"Error calculating AUC for {dataset_name}: {e}", flush=True)
        opt_ROC_AUC = np.nan

    print(f"{dataset_name} completed in {opt_time:.2f}s with AUC {opt_ROC_AUC:.4f}", flush=True)

    result_dir = os.path.join('results', dataset_name)
    ensure_dir(result_dir)

    mat_out_path = os.path.join(result_dir, f"{dataset_name}.mat")
    scipy.io.savemat(mat_out_path, {
        'opt_out_scores': opt_out_scores.reshape(-1, 1),
        'opt_ROC_AUC': opt_ROC_AUC,
        'opt_time': opt_time
    })

    xls_out_path = os.path.join(result_dir, f"{dataset_name}.xls")
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Results')

    ws.write(0, 0, 'opt_out_scores')
    ws.write(0, 1, 'opt_ROC_AUC')
    ws.write(0, 2, 'opt_time')

    for i in range(n):
        ws.write(i + 1, 0, float(opt_out_scores[i]))
        if i == 0:
            ws.write(1, 1, float(opt_ROC_AUC))
            ws.write(1, 2, float(opt_time))

    wb.save(xls_out_path)

def main():
    data_dir = 'data'
    ensure_dir('results')

    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mat') and not f.endswith('ori.mat')]
    files.sort()

    for f in files:
        process_dataset(f)

    print("All datasets processed.", flush=True)

if __name__ == '__main__':
    main()
