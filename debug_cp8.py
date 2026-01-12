# debug_cp8.py
# CP8のフレームオフセット問題をデバッグ
# Debug CP8 frame offset issue

import pandas as pd
import numpy as np

df_gt = pd.read_csv('dump_openface.csv', header=None, names=['FrameID','CP','Idx','Val'])
df_tg = pd.read_csv('dump_recover.csv', header=None, names=['FrameID','CP','Idx','Val'])

print("=" * 70)
print("CP8 Frame Offset Analysis / CP8フレームオフセット分析")
print("=" * 70)

# Get CP8 data for AU index 0 (first AU)
cp8_gt = df_gt[(df_gt['CP']=='CP8') & (df_gt['Idx']==0)].sort_values('FrameID')
cp8_tg = df_tg[(df_tg['CP']=='CP8') & (df_tg['Idx']==0)].sort_values('FrameID')

print(f"CP8 AU[0] GT frames: {len(cp8_gt)} (FrameID range: {cp8_gt['FrameID'].min()} - {cp8_gt['FrameID'].max()})")
print(f"CP8 AU[0] TG frames: {len(cp8_tg)} (FrameID range: {cp8_tg['FrameID'].min()} - {cp8_tg['FrameID'].max()})")

print("\n--- Direct comparison (same FrameID) ---")
print("FrameID | GT Value  | TG Value  | Match?")
print("-" * 45)

for frame_id in range(1, 11):  # First 10 frames
    gt_row = cp8_gt[cp8_gt['FrameID']==frame_id]
    tg_row = cp8_tg[cp8_tg['FrameID']==frame_id]
    
    gt_val = gt_row['Val'].values[0] if len(gt_row) > 0 else None
    tg_val = tg_row['Val'].values[0] if len(tg_row) > 0 else None
    
    if gt_val is not None and tg_val is not None:
        err = abs(gt_val - tg_val)
        match = "✓" if err < 0.001 else f"✗ err={err:.4f}"
        print(f"{frame_id:7} | {gt_val:9.4f} | {tg_val:9.4f} | {match}")

print("\n--- Offset comparison: GT[N] vs TG[N+1] ---")
print("仮説: TGのダンプがGTより1フレーム遅れている")
print("GT[N] | TG[N+1] | Match?")
print("-" * 45)

for frame_id in range(1, 10):  # First 9 pairs
    gt_row = cp8_gt[cp8_gt['FrameID']==frame_id]
    tg_row = cp8_tg[cp8_tg['FrameID']==frame_id+1]
    
    gt_val = gt_row['Val'].values[0] if len(gt_row) > 0 else None
    tg_val = tg_row['Val'].values[0] if len(tg_row) > 0 else None
    
    if gt_val is not None and tg_val is not None:
        err = abs(gt_val - tg_val)
        match = "✓ MATCH!" if err < 0.001 else f"✗ err={err:.4f}"
        print(f"GT[{frame_id}]={gt_val:.4f} | TG[{frame_id+1}]={tg_val:.4f} | {match}")

print("\n--- Offset comparison: GT[N+1] vs TG[N] ---")
print("仮説: TGのダンプがGTより1フレーム早い")
print("GT[N+1] | TG[N] | Match?")
print("-" * 45)

for frame_id in range(1, 10):  # First 9 pairs
    gt_row = cp8_gt[cp8_gt['FrameID']==frame_id+1]
    tg_row = cp8_tg[cp8_tg['FrameID']==frame_id]
    
    gt_val = gt_row['Val'].values[0] if len(gt_row) > 0 else None
    tg_val = tg_row['Val'].values[0] if len(tg_row) > 0 else None
    
    if gt_val is not None and tg_val is not None:
        err = abs(gt_val - tg_val)
        match = "✓ MATCH!" if err < 0.001 else f"✗ err={err:.4f}"
        print(f"GT[{frame_id+1}]={gt_val:.4f} | TG[{frame_id}]={tg_val:.4f} | {match}")

print("\n" + "=" * 70)
print("結論 / Conclusion")
print("=" * 70)
