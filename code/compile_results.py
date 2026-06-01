import os
import pandas as pd
import numpy as np
import xlwt

def main():
    base_results_path = 'results/all_outlier_result-20251025.xls'
    if not os.path.exists(base_results_path):
        print(f"Error: {base_results_path} not found.")
        return

    df = pd.read_excel(base_results_path)

    nanread_aucs = []

    for idx, row in df.iterrows():
        dataset_name = str(row['dataset'])
        dataset_result_path = os.path.join('results', dataset_name, f"{dataset_name}.xls")

        auc_score = np.nan
        if os.path.exists(dataset_result_path):
            try:
                res_df = pd.read_excel(dataset_result_path)
                if 'opt_ROC_AUC' in res_df.columns:
                    auc_score = res_df['opt_ROC_AUC'].iloc[0]
            except Exception as e:
                print(f"Error reading {dataset_result_path}: {e}")

        nanread_aucs.append(auc_score)

    if 'NaNREAD' not in df.columns:
        if 'Unnamed: 40' in df.columns:
            loc = df.columns.get_loc('Unnamed: 40')
            df.insert(loc, 'NaNREAD', nanread_aucs)
        else:
            df['NaNREAD'] = nanread_aucs
    else:
        df['NaNREAD'] = nanread_aucs

    out_path = 'results/all_outlier_result.xls'

    wb = xlwt.Workbook()
    ws = wb.add_sheet('Sheet1')

    # Write columns
    columns = list(df.columns)
    for col_idx, col_name in enumerate(columns):
        ws.write(0, col_idx, str(col_name))

    # Write data
    for row_idx, row in df.iterrows():
        for col_idx, col_name in enumerate(columns):
            val = row[col_name]
            if pd.isna(val):
                ws.write(row_idx + 1, col_idx, "")
            elif isinstance(val, (int, float, np.integer, np.floating)):
                ws.write(row_idx + 1, col_idx, float(val))
            else:
                ws.write(row_idx + 1, col_idx, str(val))

    wb.save(out_path)
    print(f"Compiled results successfully saved to {out_path}")

if __name__ == '__main__':
    main()
