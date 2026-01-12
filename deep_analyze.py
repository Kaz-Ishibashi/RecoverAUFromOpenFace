# deep_analyze.py
# Deep Analysis: Understanding Frame ID offset and HOG source difference
# 詳細分析：フレームIDオフセットとHOGソースの違いを理解する

import pandas as pd
import numpy as np

# Load dumps
df_gt = pd.read_csv('dump_openface.csv', header=None, names=['FrameID','CP','Idx','Val'])
df_tg = pd.read_csv('dump_recover.csv', header=None, names=['FrameID','CP','Idx','Val'])

print("="*70)
print("DEEP ANALYSIS: Frame ID and Data Source Investigation")
print("詳細分析：フレームIDとデータソースの調査")
print("="*70)

# ===== Frame ID Offset Analysis =====
# OpenFace FaceAnalyser::AddNextFrame does: frames_tracking++ AT THE START
# So Frame 0 in RecoverAU corresponds to Frame 1 in dump_openface?

print("\n[1] Frame ID Offset Check / フレームIDオフセット確認")
print("-"*50)

# Compare CP1 (landmarks) for Frame 0 (TG) vs Frame 1 (GT)
cp1_gt_f1 = df_gt[(df_gt['CP']=='CP1') & (df_gt['FrameID']==1)].sort_values('Idx')['Val'].values
cp1_tg_f0 = df_tg[(df_tg['CP']=='CP1') & (df_tg['FrameID']==0)].sort_values('Idx')['Val'].values

if len(cp1_gt_f1) > 0 and len(cp1_tg_f0) > 0 and len(cp1_gt_f1) == len(cp1_tg_f0):
    match = np.allclose(cp1_gt_f1, cp1_tg_f0, atol=1e-6)
    print(f"GT Frame 1 == TG Frame 0 (CP1)? {match}")
    if match:
        print(">> CONFIRMED: FrameID offset by 1!")
else:
    print(f"Length mismatch: GT F1={len(cp1_gt_f1)}, TG F0={len(cp1_tg_f0)}")

# Also check if GT Frame 0 exists
cp1_gt_f0 = df_gt[(df_gt['CP']=='CP1') & (df_gt['FrameID']==0)]
print(f"GT has Frame 0 CP1 data: {len(cp1_gt_f0) > 0}")

print("\n[2] CP3 HOG Offset Analysis / CP3 HOGオフセット分析")
print("-"*50)

# Check if GT Frame 2 HOG == TG Frame 1 HOG (after offset correction)
cp3_gt_f2 = df_gt[(df_gt['CP']=='CP3') & (df_gt['FrameID']==2)].sort_values('Idx')['Val'].values
cp3_tg_f1 = df_tg[(df_tg['CP']=='CP3') & (df_tg['FrameID']==1)].sort_values('Idx')['Val'].values

if len(cp3_gt_f2) > 0 and len(cp3_tg_f1) > 0:
    if len(cp3_gt_f2) == len(cp3_tg_f1):
        err = np.abs(cp3_gt_f2 - cp3_tg_f1)
        print(f"GT Frame 2 vs TG Frame 1 (CP3): Max Err = {err.max():.6f}, Corr = {np.corrcoef(cp3_gt_f2, cp3_tg_f1)[0,1]:.6f}")
    else:
        print(f"Length mismatch: {len(cp3_gt_f2)} vs {len(cp3_tg_f1)}")

# Check same frame comparison (after understanding offset)
cp3_gt_f1 = df_gt[(df_gt['CP']=='CP3') & (df_gt['FrameID']==1)].sort_values('Idx')['Val'].values
print(f"\nGT Frame 1 vs TG Frame 1 (same frame but different source):")
print(f"GT F1 HOG Mean: {cp3_gt_f1.mean():.6f}, TG F1 HOG Mean: {cp3_tg_f1.mean():.6f}")
if len(cp3_gt_f1) == len(cp3_tg_f1):
    corr = np.corrcoef(cp3_gt_f1, cp3_tg_f1)[0,1]
    print(f"Correlation: {corr:.6f}")

print("\n[3] HOG Value Distribution Check / HOG値分布確認")
print("-"*50)

# The key question: Is the HOG from file the same as HOG computed live?
# RecoverAU reads HOG from .hog file
# OpenFace FaceAnalyser computes HOG via Extract_FHOG_descriptor from aligned face

# If both should be identical, we need to verify the .hog file was written correctly
# and is being read correctly.

# Check value ranges
print(f"GT CP3 (F1) - Min: {cp3_gt_f1.min():.4f}, Max: {cp3_gt_f1.max():.4f}")
print(f"TG CP3 (F1) - Min: {cp3_tg_f1.min():.4f}, Max: {cp3_tg_f1.max():.4f}")

# Check if there's a systematic scaling difference
if len(cp3_gt_f1) == len(cp3_tg_f1):
    # Safely compare means instead of element-wise ratio
    gt_high = cp3_gt_f1[cp3_gt_f1 > 0.1]
    tg_high = cp3_tg_f1[cp3_tg_f1 > 0.1]
    print(f"Values > 0.1: GT count={len(gt_high)}, TG count={len(tg_high)}")
    if len(gt_high) > 0 and len(tg_high) > 0:
        print(f"Mean ratio (GT.mean / TG.mean): {gt_high.mean() / tg_high.mean():.4f}")

print("\n[4] CP4 (Geometry) Deep Dive / CP4（ジオメトリ）詳細分析")
print("-"*50)

# CP4 is constructed as: hconcat(locs.t(), geom_descriptor_frame)
# where locs = princ_comp * params_local.t()

# Check if CP4 matches after frame offset
cp4_gt_f1 = df_gt[(df_gt['CP']=='CP4') & (df_gt['FrameID']==1)].sort_values('Idx')['Val'].values
cp4_tg_f0 = df_tg[(df_tg['CP']=='CP4') & (df_tg['FrameID']==0)].sort_values('Idx')['Val'].values

if len(cp4_gt_f1) == len(cp4_tg_f0):
    err = np.abs(cp4_gt_f1 - cp4_tg_f0)
    print(f"GT Frame 1 vs TG Frame 0 (CP4): Max Err = {err.max():.6f}")
    print(f"If offset corrected, first 5 errors: {err[:5]}")
else:
    print(f"Length mismatch: {len(cp4_gt_f1)} vs {len(cp4_tg_f0)}")

print("\n[5] Summary / まとめ")
print("="*70)
print("""
KEY FINDINGS / 主要な発見:

1. Frame ID Offset: OpenFace increments frames_tracking BEFORE logging CP1.
   フレームIDオフセット: OpenFaceはCP1をログする前にframes_trackingをインクリメント。
   
2. HOG Divergence: Even with offset correction, HOG values don't match exactly.
   HOG差分: オフセット補正後もHOG値が完全には一致しない。
   
   Possible causes / 考えられる原因:
   - HOG written to file during different run vs computed live
     ファイルに書き込まれたHOGと実行中に計算されたHOGが異なる
   - Float precision loss when writing/reading .hog file (float32 vs double)
     .hogファイルの読み書き時のfloat精度損失
   - Different image preprocessing between runs
     実行間で異なる画像前処理

3. Geometry Divergence: CP4 shows large errors, likely caused by:
   ジオメトリ差分: CP4は大きな誤差を示す。原因:
   - Different PDM parameter calculation
     異なるPDMパラメータ計算
   - RecoverAU constructs geom_descriptor differently than FaceAnalyser
     RecoverAUはFaceAnalyserと異なる方法でgeom_descriptorを構築

RECOMMENDATION / 推奨:
Compare Frame-by-Frame data between the EXACT SAME RUN.
同一実行からのフレームごとのデータを比較する。

The current setup compares:
  dump_openface.csv: Generated when running FeatureExtraction.exe on video
  dump_recover.csv:  Generated when running RecoverAU.exe on .hog + .csv files

These are DIFFERENT RUNS with potentially different inputs!
これらは入力が異なる可能性のある異なる実行！
""")
