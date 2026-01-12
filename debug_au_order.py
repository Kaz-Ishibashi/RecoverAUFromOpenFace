# debug_au_order.py
# Debug AU ordering issue between OpenFace and RecoverAU
# AU順序の問題をデバッグ

import pandas as pd
import numpy as np

# Load dumps
df_gt = pd.read_csv('dump_openface.csv', header=None, names=['FrameID','CP','Idx','Val'])
df_tg = pd.read_csv('dump_recover.csv', header=None, names=['FrameID','CP','Idx','Val'])

print("=" * 70)
print("AU Ordering Debug")
print("=" * 70)

# Get CP6 data for Frame 1 (raw predictions) to see AU values
cp6_gt = df_gt[(df_gt['CP']=='CP6') & (df_gt['FrameID']==1)].sort_values('Idx')
cp6_tg = df_tg[(df_tg['CP']=='CP6') & (df_tg['FrameID']==1)].sort_values('Idx')

print(f"CP6 (Frame 1) GT count: {len(cp6_gt)}")
print(f"CP6 (Frame 1) TG count: {len(cp6_tg)}")

print("\n--- CP6 Values Comparison (Frame 1) ---")
print("This shows if the raw predictions match at each index")
print("Idx | GT Value  | TG Value  | Match?")
print("-" * 45)

for idx in range(min(len(cp6_gt), len(cp6_tg))):
    gt_row = cp6_gt[cp6_gt['Idx']==idx]
    tg_row = cp6_tg[cp6_tg['Idx']==idx]
    
    gt_val = gt_row['Val'].values[0] if len(gt_row) > 0 else None
    tg_val = tg_row['Val'].values[0] if len(tg_row) > 0 else None
    
    if gt_val is not None and tg_val is not None:
        err = abs(gt_val - tg_val)
        match = "✓" if err < 0.001 else "✗"
        print(f"{idx:3} | {gt_val:9.4f} | {tg_val:9.4f} | {match} (err={err:.4f})")

# Now compare CP7 offsets
print("\n" + "=" * 70)
print("CP7 (Offset) Analysis - Looking for pattern")
print("=" * 70)

cp7_gt = df_gt[df_gt['CP']=='CP7'].sort_values('Idx')
cp7_tg = df_tg[df_tg['CP']=='CP7'].sort_values('Idx')

# OpenFace iterates over AU_predictions_reg_all_hist which is a map (alphabetical order)
# Typical AU names: AU01, AU02, AU04, AU05, AU06, AU07, AU09, AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26, AU45
# GetAURegNames might return in a different order

print("\nHypothesis: OpenFace uses std::map which orders alphabetically")
print("RecoverAU uses GetAURegNames() which may have different order")
print("\nLet's see if the offsets match if we reorder:")

gt_vals = cp7_gt['Val'].values
tg_vals = cp7_tg['Val'].values

# Try to find matching values (regardless of position)
print(f"\nGT offsets: {gt_vals}")
print(f"TG offsets: {tg_vals}")

# Check if sorted values match
gt_sorted = sorted(gt_vals)
tg_sorted = sorted(tg_vals)
print(f"\nGT sorted: {gt_sorted}")
print(f"TG sorted: {tg_sorted}")

# Are they close when sorted?
print("\n--- Sorted value comparison ---")
for i, (g, t) in enumerate(zip(gt_sorted, tg_sorted)):
    err = abs(g - t)
    match = "✓" if err < 0.01 else "✗"
    print(f"Sorted[{i}]: GT={g:.4f}, TG={t:.4f}, err={err:.4f} {match}")
