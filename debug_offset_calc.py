# debug_offset_calc.py
# Debug offset calculation difference between OpenFace and RecoverAU
# オフセット計算の差分をデバッグ

import pandas as pd
import numpy as np

df_gt = pd.read_csv('dump_openface.csv', header=None, names=['FrameID','CP','Idx','Val'])
df_tg = pd.read_csv('dump_recover.csv', header=None, names=['FrameID','CP','Idx','Val'])

print("=" * 70)
print("Offset Calculation Debug")
print("=" * 70)

# Get CP6 (raw predictions) for analysis
cp6_gt = df_gt[df_gt['CP']=='CP6'].sort_values(['FrameID', 'Idx'])
cp6_tg = df_tg[df_tg['CP']=='CP6'].sort_values(['FrameID', 'Idx'])

# Get CP7 (offsets)
cp7_gt = df_gt[df_gt['CP']=='CP7'].sort_values('Idx')
cp7_tg = df_tg[df_tg['CP']=='CP7'].sort_values('Idx')

print(f"CP6 GT unique indices: {sorted(cp6_gt['Idx'].unique())}")
print(f"CP6 TG unique indices: {sorted(cp6_tg['Idx'].unique())}")
print(f"CP7 GT count: {len(cp7_gt)}, indices: {list(cp7_gt['Idx'].values)}")
print(f"CP7 TG count: {len(cp7_tg)}, indices: {list(cp7_tg['Idx'].values)}")

# The key insight:
# Both CP6 and CP7 dump with indices 0-16 (17 AUs total)
# CP6 passed, so raw_reg values are correct
# But CP7 values differ

# Let's manually calculate what the offsets SHOULD be
# by collecting all CP6 values for each AU index

print("\n--- Manual Offset Calculation from CP6 ---")
print("For each AU, collect all valid (frame > 0) predictions, sort, and find cutoff")

# Get all CP6 values per AU
for au_idx in range(17):
    gt_preds = cp6_gt[cp6_gt['Idx']==au_idx]['Val'].values
    tg_preds = cp6_tg[cp6_tg['Idx']==au_idx]['Val'].values
    
    gt_sorted = sorted(gt_preds)
    tg_sorted = sorted(tg_preds)
    
    # Get corresponding CP7 offset
    gt_offset_row = cp7_gt[cp7_gt['Idx']==au_idx]
    tg_offset_row = cp7_tg[cp7_tg['Idx']==au_idx]
    
    gt_offset = gt_offset_row['Val'].values[0] if len(gt_offset_row) > 0 else None
    tg_offset = tg_offset_row['Val'].values[0] if len(tg_offset_row) > 0 else None
    
    if au_idx < 5:  # Just first 5 for brevity
        print(f"\nAU Index {au_idx}:")
        print(f"  GT preds: {len(gt_preds)}, min={min(gt_preds):.4f}, max={max(gt_preds):.4f}")
        print(f"  TG preds: {len(tg_preds)}, min={min(tg_preds):.4f}, max={max(tg_preds):.4f}")
        gt_str2 = f"{gt_offset:.4f}" if gt_offset is not None else "N/A"
        tg_str2 = f"{tg_offset:.4f}" if tg_offset is not None else "N/A"
        print(f"  GT offset: {gt_str2}")
        print(f"  TG offset: {tg_str2}")
        
        # Check if predictions match
        if len(gt_preds) == len(tg_preds):
            max_diff = max(abs(a-b) for a, b in zip(sorted(gt_preds), sorted(tg_preds)))
            print(f"  Max diff in sorted preds: {max_diff:.6f}")

print("\n--- CP7 Full Comparison ---")
for au_idx in range(17):
    gt_row = cp7_gt[cp7_gt['Idx']==au_idx]
    tg_row = cp7_tg[cp7_tg['Idx']==au_idx]
    gt_val = gt_row['Val'].values[0] if len(gt_row) > 0 else None
    tg_val = tg_row['Val'].values[0] if len(tg_row) > 0 else None
    match = "✓" if gt_val is not None and tg_val is not None and abs(gt_val - tg_val) < 0.001 else "✗"
    gt_str = f"{gt_val:.4f}" if gt_val is not None else "N/A"
    tg_str = f"{tg_val:.4f}" if tg_val is not None else "N/A"
    print(f"AU {au_idx:2}: GT={gt_str}, TG={tg_str} {match}")
