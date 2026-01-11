# analyze_dumps.py
# Dump Analysis Script for understanding CP3 (HOG) divergence
# CP3ダンプ解析スクリプト：HOG差分の原因特定

import pandas as pd
import numpy as np

# Load dumps / ダンプ読み込み
df_gt = pd.read_csv('dump_openface.csv', header=None, names=['FrameID','CP','Idx','Val'])
df_tg = pd.read_csv('dump_recover.csv', header=None, names=['FrameID','CP','Idx','Val'])

print("="*60)
print("CHECKPOINT ANALYSIS / チェックポイント分析")
print("="*60)

# Check CP1 (landmarks) difference / CP1（ランドマーク）差分確認
cp1_gt = df_gt[df_gt['CP']=='CP1'].sort_values(['FrameID', 'Idx'])
cp1_tg = df_tg[df_tg['CP']=='CP1'].sort_values(['FrameID', 'Idx'])

if len(cp1_gt) == len(cp1_tg):
    cp1_err = np.abs(cp1_gt['Val'].values - cp1_tg['Val'].values)
    print(f"CP1 (Landmarks) Max Error: {cp1_err.max():.2e}")
    print(f"CP1 Match (tol=1e-6): {cp1_err.max() < 1e-6}")
else:
    print(f"CP1 Length Mismatch: GT={len(cp1_gt)}, TG={len(cp1_tg)}")

print("-"*60)

# CP3 (HOG) Analysis / CP3（HOG）分析
cp3_gt = df_gt[df_gt['CP']=='CP3'].sort_values(['FrameID', 'Idx'])
cp3_tg = df_tg[df_tg['CP']=='CP3'].sort_values(['FrameID', 'Idx'])

print(f"CP3 (HOG) GT count: {len(cp3_gt)}, TG count: {len(cp3_tg)}")
print(f"CP3 GT Frame IDs: {sorted(cp3_gt['FrameID'].unique())}")
print(f"CP3 TG Frame IDs: {sorted(cp3_tg['FrameID'].unique())}")

# Merge for detailed comparison
cp3_merged = pd.merge(cp3_gt, cp3_tg, on=['FrameID', 'Idx'], suffixes=('_GT','_TG'), how='inner')
if not cp3_merged.empty:
    cp3_merged['AbsErr'] = np.abs(cp3_merged['Val_GT'] - cp3_merged['Val_TG'])
    print(f"CP3 Merged rows: {len(cp3_merged)}")
    print(f"CP3 Max Abs Error: {cp3_merged['AbsErr'].max():.6f}")
    print(f"CP3 Mean Abs Error: {cp3_merged['AbsErr'].mean():.6f}")
    
    # Show first few mismatches
    errors = cp3_merged[cp3_merged['AbsErr'] > 1e-6].head(5)
    if not errors.empty:
        print("\nFirst CP3 Errors:")
        print(errors.to_string(index=False))
        
    # Check if values are systematically shifted
    print("\n--- Frame 1 Correlation Check ---")
    frame1_gt = cp3_merged[cp3_merged['FrameID']==1]['Val_GT'].values
    frame1_tg = cp3_merged[cp3_merged['FrameID']==1]['Val_TG'].values
    if len(frame1_gt) > 0:
        corr = np.corrcoef(frame1_gt, frame1_tg)[0,1]
        print(f"Frame 1 Correlation: {corr:.6f}")
        print(f"Frame 1 GT Mean: {frame1_gt.mean():.6f}, Std: {frame1_gt.std():.6f}")
        print(f"Frame 1 TG Mean: {frame1_tg.mean():.6f}, Std: {frame1_tg.std():.6f}")
        
        # Check if values are in completely different order (same values, different indices)
        print("\n--- Order Check (Same values, different order?) ---")
        gt_sorted = np.sort(frame1_gt)
        tg_sorted = np.sort(frame1_tg)
        sorted_match = np.allclose(gt_sorted, tg_sorted, atol=1e-3)
        print(f"Sorted values match (tol=1e-3): {sorted_match}")

print("-"*60)

# CP4 (Geometry) Analysis
cp4_gt = df_gt[df_gt['CP']=='CP4'].sort_values(['FrameID', 'Idx'])
cp4_tg = df_tg[df_tg['CP']=='CP4'].sort_values(['FrameID', 'Idx'])
print(f"CP4 (Geometry) GT count: {len(cp4_gt)}, TG count: {len(cp4_tg)}")

if len(cp4_gt) > 0 and len(cp4_tg) > 0:
    cp4_merged = pd.merge(cp4_gt, cp4_tg, on=['FrameID', 'Idx'], suffixes=('_GT','_TG'), how='inner')
    cp4_merged['AbsErr'] = np.abs(cp4_merged['Val_GT'] - cp4_merged['Val_TG'])
    print(f"CP4 Max Abs Error: {cp4_merged['AbsErr'].max():.6f}")
    errors = cp4_merged[cp4_merged['AbsErr'] > 1e-6].head(3)
    if not errors.empty:
        print("\nFirst CP4 Errors:")
        print(errors.to_string(index=False))

print("-"*60)

# CP6 (Raw Prediction) Analysis
cp6_gt = df_gt[df_gt['CP']=='CP6'].sort_values(['FrameID', 'Idx'])
cp6_tg = df_tg[df_tg['CP']=='CP6'].sort_values(['FrameID', 'Idx'])
print(f"CP6 (Raw Prediction) GT count: {len(cp6_gt)}, TG count: {len(cp6_tg)}")

if len(cp6_gt) > 0 and len(cp6_tg) > 0:
    cp6_merged = pd.merge(cp6_gt, cp6_tg, on=['FrameID', 'Idx'], suffixes=('_GT','_TG'), how='inner')
    cp6_merged['AbsErr'] = np.abs(cp6_merged['Val_GT'] - cp6_merged['Val_TG'])
    print(f"CP6 Max Abs Error: {cp6_merged['AbsErr'].max():.6f}")
    
    # Frame 0 comparison
    frame0_cp6 = cp6_merged[cp6_merged['FrameID']==0]
    if not frame0_cp6.empty:
        print("\nFrame 0 CP6 (First 5 AUs):")
        print(frame0_cp6.head().to_string(index=False))

print("="*60)
print("CONCLUSION / 結論")
print("="*60)
