import argparse
import numpy as np
import scipy.io
import sys
from code.nanread import NaNREAD

def load_mat(filepath):
    try:
        return scipy.io.loadmat(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run NaNREAD algorithm on a dataset.")
    parser.add_argument('mat_file', type=str, help='Path to the .mat file')
    parser.add_argument('--variable', type=str, default=None, help='Variable name in the .mat file to process')

    args = parser.parse_args()

    data_dict = load_mat(args.mat_file)
    if data_dict is None:
        sys.exit(1)

    data = None
    if args.variable:
        if args.variable in data_dict:
            data = data_dict[args.variable]
        else:
            print(f"Variable '{args.variable}' not found in the MAT file.")
            sys.exit(1)
    else:
        for key, val in data_dict.items():
            if not key.startswith('__') and hasattr(val, 'shape'):
                print(f"Auto-selected variable '{key}' with shape {val.shape}")
                data = np.array(val)
                break

    if data is None:
        print("Could not find a valid variable in the file.")
        sys.exit(1)

    print(f"Running NaNREAD on data of shape {data.shape}...")
    try:
        scores = NaNREAD(data)
        print("\n=== Anomaly Scores (NaNREAF) ===")
        for i, score in enumerate(scores):
            print(f"Object {i+1}: {score:.4f}")
    except Exception as e:
        print(f"An error occurred during algorithm execution: {e}")

if __name__ == "__main__":
    main()
