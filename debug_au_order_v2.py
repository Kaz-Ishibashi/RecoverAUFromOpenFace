# debug_au_order_v2.py
# Analyze AU ordering difference between OpenFace and RecoverAU for CP7
# CP7のOpenFaceとRecoverAU間のAU順序の差分を分析

import pandas as pd
import numpy as np

df_gt = pd.read_csv('dump_openface.csv', header=None, names=['FrameID','CP','Idx','Val'])
df_tg = pd.read_csv('dump_recover.csv', header=None, names=['FrameID','CP','Idx','Val'])

print("=" * 70)
print("CP7 AU Ordering Analysis")
print("=" * 70)

# Get CP7 data
cp7_gt = df_gt[df_gt['CP']=='CP7'].sort_values('Idx')
cp7_tg = df_tg[df_tg['CP']=='CP7'].sort_values('Idx')

print(f"GT CP7 count: {len(cp7_gt)}")
print(f"TG CP7 count: {len(cp7_tg)}")

print("\n--- Direct Index Comparison ---")
print("Idx | GT Value  | TG Value")
print("-" * 35)

all_indices = sorted(set(cp7_gt['Idx'].tolist() + cp7_tg['Idx'].tolist()))
for idx in all_indices:
    gt_row = cp7_gt[cp7_gt['Idx']==idx]
    tg_row = cp7_tg[cp7_tg['Idx']==idx]
    gt_val = gt_row['Val'].values[0] if len(gt_row) > 0 else "N/A"
    tg_val = tg_row['Val'].values[0] if len(tg_row) > 0 else "N/A"
    gt_str = f"{gt_val:.4f}" if isinstance(gt_val, float) else gt_val
    tg_str = f"{tg_val:.4f}" if isinstance(tg_val, float) else tg_val
    print(f"{idx:3} | {gt_str:9} | {tg_str:9}")

print("\n--- Looking for matching values (regardless of index) ---")
gt_vals = sorted([v for v in cp7_gt['Val'].values if v > 0])
tg_vals = sorted([v for v in cp7_tg['Val'].values if v > 0])

print(f"GT non-zero sorted: {[f'{v:.4f}' for v in gt_vals]}")
print(f"TG non-zero sorted: {[f'{v:.4f}' for v in tg_vals]}")

# OpenFace uses AU_predictions_reg_all_hist which is a map (ordered alphabetically)
# RecoverAU uses GetAURegNames() which returns a vector
# The map has AU names as keys, so it's sorted alphabetically like:
# AU01, AU02, AU04, AU05, AU06, AU07, AU09, AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26, AU45

# GetAURegNames() might return in the order they were added, which could be different

print("\n--- Hypothesis: Different AU ordering ---")
print("""
OpenFace iteration: AU_predictions_reg_all_hist (std::map)
  -> Alphabetical order: AU01, AU02, AU04, AU05, AU06, ...

RecoverAU iteration: GetAURegNames()
  -> Order depends on how AUs are added to the vector

If orders differ, the same index will refer to different AUs!
""")
