import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import argparse
import os
import sys

# def compare_with_ground_truth(gt_path, recovered_path):
#     """
#     Ground Truth CSVと復元CSVを比較評価する
#     """
#     print(f"--- Ground Truth Comparison ---")
#     print(f"GT: {gt_path}")
#     print(f"Recovered: {recovered_path}")
# 
#     try:
#         df_gt = pd.read_csv(gt_path, skipinitialspace=True)
#         df_rec = pd.read_csv(recovered_path, skipinitialspace=True)
#     except Exception as e:
#         print(f"Error reading CSV files: {e}")
#         return
# 
#     # カラム名のクリーニング（スペース削除など）
#     df_gt.columns = df_gt.columns.str.strip()
#     df_rec.columns = df_rec.columns.str.strip()
# 
#     # マージキーの決定（timestampが存在すれば優先、なければframe）
#     merge_key = 'timestamp' if 'timestamp' in df_gt.columns and 'timestamp' in df_rec.columns else 'frame'
#     
#     # 復元側が success=1 の行のみ対象にするなどのフィルタリングが必要ならここで行う
#     # 今回はマージ後にフィルタリングする方針とする
# 
#     # マージ
#     # OpenFaceの出力形式に合わせて suffix を調整
#     merged = pd.merge(df_gt, df_rec, on=merge_key, suffixes=('_gt', '_rec'), how='inner')
# 
#     # success カラムがあればフィルタリング
#     if 'success' in merged.columns:
#         valid_data = merged[merged['success'] == 1]
#         print(f"Valid frames (success=1 in recovered): {len(valid_data)} / {len(merged)}")
#     else:
#         valid_data = merged
#         print(f"Total matched frames: {len(valid_data)}")
# 
#     if len(valid_data) == 0:
#         print("No valid overlapping data found.")
#         return
# 
#     au_columns = [col.replace('_rec', '') for col in valid_data.columns if col.startswith('AU') and col.endswith('_r_rec')]
#     # 'AU01_r' のような形式に戻す
#     
#     results = []
#     
#     print("\n[Evaluation Results]")
#     print(f"{'AU Code':<10} | {'MAE':<10} | {'Correlation':<12} | {'Perfect Match (>0.99)':<20}")
#     print("-" * 60)
# 
#     for au_col in au_columns:
#         base_name = au_col  # AU01_r
#         gt_col = base_name + '_gt'
#         rec_col = base_name + '_rec' # AU01_r_rec
# 
#         if gt_col not in valid_data.columns:
#             continue
# 
#         y_true = valid_data[gt_col]
#         y_pred = valid_data[rec_col]
# 
#         mae = np.mean(np.abs(y_true - y_pred))
#         
#         # 標準偏差が0の場合は相関が定義できないためNaNまたは0扱い
#         if y_true.std() == 0 or y_pred.std() == 0:
#             corr = 0.0
#         else:
#             corr, _ = stats.pearsonr(y_true, y_pred)
# 
#         is_perfect = "YES" if corr > 0.99 else "NO"
#         
#         print(f"{base_name:<10} | {mae:<10.4f} | {corr:<12.4f} | {is_perfect:<20}")
#         
#         results.append({
#             'AU': base_name,
#             'MAE': mae,
#             'Corr': corr
#         })

def check_consistency_with_landmarks(recovered_path, landmarks_path, output_dir):
    """
    復元データとランドマークの物理的整合性をチェックする
    """
    print(f"\n--- Consistency Check with Landmarks ---")
    print(f"Recovered: {recovered_path}")
    print(f"Landmarks: {landmarks_path}")

    try:
        df_rec = pd.read_csv(recovered_path, skipinitialspace=True)
        # ランドマークCSVが別ファイルの場合のみ読み込む。同じ場合はdf_recを使う
        if recovered_path == landmarks_path:
            df_lm = df_rec
        else:
            df_lm = pd.read_csv(landmarks_path, skipinitialspace=True)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    df_rec.columns = df_rec.columns.str.strip()
    df_lm.columns = df_lm.columns.str.strip()

    # マージ
    merge_key = 'timestamp' if 'timestamp' in df_rec.columns and 'timestamp' in df_lm.columns else 'frame'
    
    # suffixをつける
    merged = pd.merge(df_rec, df_lm, on=merge_key, suffixes=('', '_lm'), how='inner')
    
    # ランドマークカラム名が x_0, y_0 ... or X_0, Y_0 ... の形式か確認
    # OpenFace 2.2.0 output format check
    # 通常は x_0, y_0 ... (2D) または X_0, Y_0 ... (3D)
    # ここでは 2D landmarks (x_#, y_#) を使用する
    
    def get_dist(df, idx1, idx2):
        x1, y1 = df[f'x_{idx1}'], df[f'y_{idx1}']
        x2, y2 = df[f'x_{idx2}'], df[f'y_{idx2}']
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    # 必要なカラムがあるかチェック
    required_lms = [48, 54, 37, 41, 38, 40, 43, 47, 44, 46]
    for idx in required_lms:
        if f'x_{idx}' not in merged.columns or f'y_{idx}' not in merged.columns:
            print(f"Error: Landmark columns x_{idx}, y_{idx} not found.")
            # カラム名探索: 3Dの場合 X_0, Y_0 かもしれないのでチェック
            if f'X_{idx}' in merged.columns:
                 # カラム名置換などの対応も可能だが今回は簡易的にエラー
                 print("Hint: Only found X_#, Y_# columns? This script uses x_#, y_# (2D).")
            return

    # 1. Smile Check (AU12 vs Lip Corner Distance)
    # AU12: Lip Corner Puller -> 口角が広がる -> 距離増大
    lip_dist = get_dist(merged, 48, 54)
    au12 = merged['AU12_r']
    
    corr_smile, _ = stats.pearsonr(au12, lip_dist)
    print(f"\n[Smile Check] AU12 vs Mouth Width ({len(merged)} frames)")
    print(f"Correlation: {corr_smile:.4f}")
    if corr_smile > 0.5:
        print("Result: OK (Positive correlation detected)")
    else:
        print("Result: WARNING (Low or negative correlation)")

    # 2. Blink Check (AU45 vs Eye Opening)
    # AU45: Blink -> 目が閉じる -> 距離減少 -> 負の相関
    left_eye_open = (get_dist(merged, 37, 41) + get_dist(merged, 38, 40)) / 2
    right_eye_open = (get_dist(merged, 43, 47) + get_dist(merged, 44, 46)) / 2
    avg_eye_open = (left_eye_open + right_eye_open) / 2
    au45 = merged['AU45_r']

    corr_blink, _ = stats.pearsonr(au45, avg_eye_open)
    print(f"\n[Blink Check] AU45 vs Avg Eye Opening")
    print(f"Correlation: {corr_blink:.4f}")
    if corr_blink < -0.5:
        print("Result: OK (Negative correlation detected)")
    else:
        print("Result: WARNING (Weak or positive correlation)")

    # 3. Visualization
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))
        
        # 正規化してプロットしやすくする
        def normalize(series):
            return (series - series.min()) / (series.max() - series.min())
            
        plt.plot(merged[merge_key], normalize(au45), label='AU45_r (Blink)', alpha=0.7)
        plt.plot(merged[merge_key], normalize(avg_eye_open), label='Eye Opening (Landmark)', alpha=0.7)
        
        plt.title('Consistency Check: AU45 vs Eye Opening')
        plt.xlabel(merge_key)
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(output_dir, 'blink_consistency.png')
        plt.savefig(save_path)
        print(f"\nPlot saved to: {save_path}")

def plot_discrepancy(merged_df, au_col, gt_col, rec_col, output_dir):
    """
    AU値の比較プロットを作成する
    """
    plt.figure(figsize=(12, 6))
    
    # 時系列プロット
    plt.subplot(2, 1, 1)
    plt.plot(merged_df.index, merged_df[gt_col], label='Ground Truth', alpha=0.7)
    plt.plot(merged_df.index, merged_df[rec_col], label='Recovered', alpha=0.7)
    plt.title(f'Comparison: {au_col}')
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    
    # 散布図
    plt.subplot(2, 1, 2)
    plt.scatter(merged_df[gt_col], merged_df[rec_col], alpha=0.5, s=10)
    
    # 理想線 (y=x)
    min_val = min(merged_df[gt_col].min(), merged_df[rec_col].min())
    max_val = max(merged_df[gt_col].max(), merged_df[rec_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
    plt.xlabel('Ground Truth')
    plt.ylabel('Recovered')
    plt.title(f'Scatter Plot: {au_col}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'discrepancy_{au_col}.png')
    plt.savefig(save_path)
    plt.close()

def compare_with_ground_truth(gt_path, recovered_path, output_dir=None):
    """
    Ground Truth CSVと復元CSVを比較評価する
    """
    print(f"--- Ground Truth Comparison ---")
    print(f"GT: {gt_path}")
    print(f"Recovered: {recovered_path}")

    try:
        df_gt = pd.read_csv(gt_path, skipinitialspace=True)
        df_rec = pd.read_csv(recovered_path, skipinitialspace=True)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # カラム名のクリーニング（スペース削除など）
    df_gt.columns = df_gt.columns.str.strip()
    df_rec.columns = df_rec.columns.str.strip()

    # Merge Strategy: Use 'frame' column instead of timestamp to avoid float precision issues.
    # Note: OpenFace GT is usually 1-indexed, RecoverAU is 0-indexed.
    # Check and align.
    
    # Detect frame text/int
    df_gt['frame'] = df_gt['frame'].astype(int)
    df_rec['frame'] = df_rec['frame'].astype(int)
    
    if df_gt['frame'].min() == 1 and df_rec['frame'].min() == 0:
        print("Detected 0-based vs 1-based frame indexing. Aligning Recovered (0->1) to match GT.")
        df_rec['frame'] = df_rec['frame'] + 1
    
    merge_key = 'frame'
    print(f"Merging on key: {merge_key}")

    # OpenFaceの出力形式に合わせて suffix を調整
    merged = pd.merge(df_gt, df_rec, on=merge_key, suffixes=('_gt', '_rec'), how='inner')

    # success_rec カラムがあればフィルタリング
    if 'success_rec' in merged.columns:
        valid_data = merged[merged['success_rec'] == 1]
        print(f"Valid frames (success=1 in recovered): {len(valid_data)} / {len(merged)}")
    elif 'success' in merged.columns and 'success_rec' not in merged.columns:
         if 'success_rec' in merged.columns:
             valid_data = merged[merged['success_rec'] == 1]
         elif 'success' in merged.columns: # default 'success' might come from GT if no suffix applied to join key? actually merge suffixes applies to overlapping columns.
             # check which success column to use.
             # usually success_gt and success_rec
             if 'success_rec' in merged.columns:
                 valid_data = merged[merged['success_rec'] == 1]
             elif 'success_gt' in merged.columns:
                 # Fallback to GT success as proxy? Better to be safe.
                 valid_data = merged[merged['success_gt'] == 1]
             else:
                 valid_data = merged
         else:
             valid_data = merged
         print(f"Valid frames (inferred success): {len(valid_data)} / {len(merged)}")
    else:
        # Check specifically for _rec and _gt success columns after merge
        if 'success_rec' in merged.columns:
            valid_data = merged[merged['success_rec'] == 1]
        elif 'success_gt' in merged.columns:
             valid_data = merged[merged['success_gt'] == 1]
        else:
            valid_data = merged
        print(f"Total matched frames (checking specific success cols): {len(valid_data)}")


    if len(valid_data) == 0:
        print("No valid overlapping data found.")
        return

    au_r_columns = sorted([col.replace('_rec', '') for col in valid_data.columns if col.startswith('AU') and col.endswith('_r_rec')])
    au_c_columns = sorted([col.replace('_rec', '') for col in valid_data.columns if col.startswith('AU') and col.endswith('_c_rec')])
    
    results = []
    
    if au_r_columns:
        print("\n[Intensity Evaluation Results (Success Only)]")
        print(f"{'AU Code':<10} | {'MAE':<10} | {'Correlation':<12} | {'Perfect Match (>0.9)':<20}")
        print("-" * 65)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for base_name in au_r_columns:
            gt_col = base_name + '_gt'
            rec_col = base_name + '_rec'

            if gt_col not in valid_data.columns:
                continue

            y_true = valid_data[gt_col]
            y_pred = valid_data[rec_col]

            mae = np.mean(np.abs(y_true - y_pred))
            
            # 標準偏差が0の場合は相関が定義できないためNaNまたは0扱い
            if y_true.std() == 0 or y_pred.std() == 0:
                corr = 0.0
            else:
                corr, _ = stats.pearsonr(y_true, y_pred)

            is_perfect = "YES" if corr > 0.9 else "NO"
            
            print(f"{base_name:<10} | {mae:<10.4f} | {corr:<12.4f} | {is_perfect:<20}")
            
            results.append({
                'AU': base_name,
                'Type': 'Intensity',
                'MAE': mae,
                'Corr': corr
            })

            if output_dir:
                plot_discrepancy(valid_data, base_name, gt_col, rec_col, output_dir)

    if au_c_columns:
        print("\n[Classification Evaluation Results (Success Only)]")
        print(f"{'AU Code':<10} | {'Accuracy':<10} | {'F1-Score':<10}")
        print("-" * 40)

        for base_name in au_c_columns:
            gt_col = base_name + '_gt'
            rec_col = base_name + '_rec'

            if gt_col not in valid_data.columns:
                continue

            y_true = valid_data[gt_col].astype(int)
            y_pred = valid_data[rec_col].astype(int)

            accuracy = np.mean(y_true == y_pred)
            
            # 簡易的なF1-Score計算 (sklearnを使わない場合)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"{base_name:<10} | {accuracy:<10.4f} | {f1:<10.4f}")
            
            results.append({
                'AU': base_name,
                'Type': 'Classification',
                'Accuracy': accuracy,
                'F1': f1
            })
    
    return results if 'results' in locals() else []

def check_pipeline_pass(results, min_corr=0.99, max_mae=0.1):
    """
    Check if the results meet the pass criteria.
    """
    print(f"\n[Pipeline Pass Check] Criteria: Min Corr >= {min_corr}, Max MAE <= {max_mae}")
    
    all_pass = True
    failed_aus = []

    for res in results:
        if res['Type'] == 'Intensity':
            au = res['AU']
            mae = res['MAE']
            corr = res['Corr']
            
            # Skip correlation check if std dev was 0 (corr=0) but MAE is very low (flatline match)
            # MAE check is primary.
            if mae > max_mae:
                all_pass = False
                failed_aus.append(f"{au} (MAE {mae:.4f} > {max_mae})")
            elif corr < min_corr and mae > 1e-5: # If MAE is tiny, correlation might be noise
                all_pass = False
                failed_aus.append(f"{au} (Corr {corr:.4f} < {min_corr})")
    
    if all_pass:
        print("[SUCCESS] Pipeline verification PASSED for all AUs.")
        return True
    else:
        print(f"[FAIL] Pipeline verification FAILED for: {', '.join(failed_aus)}")
        return False



def main():
    parser = argparse.ArgumentParser(description='Verify AU Recovery Results')
    subparsers = parser.add_subparsers(dest='command', help='Verification mode')

    # Mode 1: Ground Truth Comparison
    parser_gt = subparsers.add_parser('compare', help='Compare with Ground Truth')
    parser_gt.add_argument('--gt', required=True, help='Path to Ground Truth CSV')
    parser_gt.add_argument('--rec', required=True, help='Path to Recovered CSV')
    parser_gt.add_argument('--out', help='Output directory for discrepancy plots')
    parser_gt.add_argument('--tolerance', type=float, default=0.1, help='Max MAE tolerance')
    parser_gt.add_argument('--min_corr', type=float, default=0.99, help='Min Correlation tolerance')

    # Mode 2: Consistency Check
    parser_cons = subparsers.add_parser('consistency', help='Check consistency with physics/landmarks')
    parser_cons.add_argument('--rec', required=True, help='Path to Recovered CSV')
    parser_cons.add_argument('--landmarks', required=True, help='Path to Landmarks CSV (can be same as rec)')
    parser_cons.add_argument('--out', required=True, help='Output directory for plots')

    args = parser.parse_args()

    if args.command == 'compare':
        results = compare_with_ground_truth(args.gt, args.rec, args.out)
        if results:
            success = check_pipeline_pass(results, min_corr=args.min_corr, max_mae=args.tolerance)
            sys.exit(0 if success else 1)
        else:
            sys.exit(1) # No data or failure

    elif args.command == 'consistency':
        check_consistency_with_landmarks(args.rec, args.landmarks, args.out)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
