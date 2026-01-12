# debug_cp7.py
# Debug CP7 (Offset/Cutoff) differences between OpenFace and RecoverAU
# CP7（オフセット/カットオフ）のOpenFaceとRecoverAU間の差分をデバッグ

import pandas as pd
import numpy as np

# Load dumps
df_gt = pd.read_csv('dump_openface.csv', header=None, names=['FrameID','CP','Idx','Val'])
df_tg = pd.read_csv('dump_recover.csv', header=None, names=['FrameID','CP','Idx','Val'])

print("=" * 70)
print("CP7 (Offset/Cutoff) Debug Analysis")
print("=" * 70)

# Get CP7 data
cp7_gt = df_gt[df_gt['CP']=='CP7'].sort_values(['Idx'])
cp7_tg = df_tg[df_tg['CP']=='CP7'].sort_values(['Idx'])

print(f"CP7 GT rows: {len(cp7_gt)}")
print(f"CP7 TG rows: {len(cp7_tg)}")
print(f"CP7 GT FrameIDs: {cp7_gt['FrameID'].unique()}")
print(f"CP7 TG FrameIDs: {cp7_tg['FrameID'].unique()}")

# Compare values
print("\n--- CP7 Comparison (Offset per AU) ---")
print("Idx | AU    | GT_Offset | TG_Offset | Match?")
print("-" * 55)

# Merge on Idx (AU index)
merged = pd.merge(cp7_gt, cp7_tg, on=['Idx'], suffixes=('_GT', '_TG'), how='outer')
merged['AbsErr'] = np.abs(merged['Val_GT'].fillna(0) - merged['Val_TG'].fillna(0))

for _, row in merged.iterrows():
    idx = int(row['Idx'])
    gt_val = row['Val_GT'] if pd.notna(row['Val_GT']) else None
    tg_val = row['Val_TG'] if pd.notna(row['Val_TG']) else None
    err = row['AbsErr']
    match = "✓" if err < 0.001 else "✗"
    
    gt_str = f"{gt_val:.6f}" if gt_val is not None else "N/A"
    tg_str = f"{tg_val:.6f}" if tg_val is not None else "N/A"
    
    print(f"{idx:3} |       | {gt_str:12} | {tg_str:12} | {match} (err={err:.4f})")

print("\n" + "=" * 70)
print("Analysis of Failures")
print("=" * 70)

# Check which AUs have non-zero GT but zero TG
failures = merged[merged['AbsErr'] > 0.001]
print(f"\nNumber of failing AUs: {len(failures)}")

print("\nFailing AUs where GT is non-zero but TG is zero (or very different):")
for _, row in failures.iterrows():
    print(f"  AU Index {int(row['Idx'])}: GT={row['Val_GT']:.4f}, TG={row['Val_TG']:.4f}")

print("\n" + "=" * 70)
print("Hypothesis Check")
print("=" * 70)
print("""
The offset (cutoff) is calculated in ExtractAllPredictionsOfflineReg as:
1. Collect all valid predictions for this AU
2. Sort them
3. Use the cutoff ratio to find the baseline
4. offset = predictions[size * cutoff_ratio]

Possible issues:
- Different AU ordering between GT and TG
- Different cutoff ratios being used
- Different set of AUs being marked as "dynamic"
- RecoverAU might not be including all frames in offset calculation
""")
