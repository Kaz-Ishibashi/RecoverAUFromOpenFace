# verify_pipeline.py
# Pipeline Verification Script with Per-Checkpoint Tolerances
# パイプライン検証スクリプト (チェックポイント別許容値対応)
#
# Changelog:
# - v2: Added per-checkpoint tolerances (CP3/CP5/CP6/CP7/CP8 use 1e-3 for float32 precision)
# - v1: Initial version with single tolerance

import pandas as pd
import numpy as np
import sys
import os

# Per-Checkpoint Tolerances / チェックポイント別許容値
# CP1, CP2, CP4: Exact match expected (1e-6)
# CP3, CP5_HOG, CP5_Geom: float32 precision loss from .hog file I/O (1e-3)
# CP6, CP7, CP8: Derived from CP3/CP5, inherit precision loss (1e-3)
TOLERANCE_BY_CP = {
    'CP1': 1e-6,      # Raw Landmarks - exact match
    'CP2': 1e-6,      # Aligned Face Sum - exact match
    'CP3': 1e-3,      # HOG Descriptor - float32 precision loss
    'CP4': 1e-6,      # Geometry Descriptor - exact match
    'CP5_HOG': 1e-3,  # HOG Median - inherits from CP3
    'CP5_Geom': 1e-6, # Geometry Median - exact match
    'CP6': 1e-3,      # Raw Prediction - inherits from CP3
    'CP7': 1e-3,      # Offset - inherits from CP6
    'CP8': 1e-3,      # Final Value - inherits from CP6
}

# Default tolerance for undefined checkpoints
DEFAULT_TOLERANCE = 1e-3


def get_tolerance(checkpoint_id):
    """
    Returns the appropriate tolerance for a given checkpoint.
    チェックポイントに応じた許容値を返す。
    """
    return TOLERANCE_BY_CP.get(checkpoint_id, DEFAULT_TOLERANCE)


def verify_pipeline(openface_csv, recover_csv):
    """
    Verifies that the values in recover_csv match openface_csv within per-checkpoint tolerances.
    
    openface_csv: Path to Ground Truth CSV (from FaceAnalyser.cpp)
    recover_csv: Path to Target CSV (from RecoverAU.cpp)
    
    recover_csvがopenface_csvの値とチェックポイント別許容誤差以内で一致することを検証します。
    """
    
    print(f"Loading Ground Truth: {openface_csv}")
    try:
        df_gt = pd.read_csv(openface_csv, header=None, names=['FrameID', 'CheckpointID', 'DimIndex', 'Value'])
    except Exception as e:
        print(f"Error loading {openface_csv}: {e}")
        return False

    print(f"Loading Target: {recover_csv}")
    try:
        df_target = pd.read_csv(recover_csv, header=None, names=['FrameID', 'CheckpointID', 'DimIndex', 'Value'])
    except Exception as e:
        print(f"Error loading {recover_csv}: {e}")
        return False

    # Check for empty data
    if df_gt.empty:
        print("Error: Ground Truth CSV is empty.")
        return False
    if df_target.empty:
        print("Error: Target CSV is empty.")
        return False

    # Merge dataframes on keys
    # キーに基づいてデータフレームをマージ
    print("Merging data...")
    merged = pd.merge(df_gt, df_target, on=['FrameID', 'CheckpointID', 'DimIndex'], suffixes=('_GT', '_Target'), how='inner')
    
    if merged.empty:
        print("Error: No matching FrameID/CheckpointID/DimIndex found between files.")
        return False

    print(f"Merged {len(merged)} rows for comparison.\n")

    # Calculate Error with Per-Checkpoint Tolerance
    # チェックポイント別許容値で誤差計算
    merged['AbsError'] = (merged['Value_GT'] - merged['Value_Target']).abs()
    merged['Tolerance'] = merged['CheckpointID'].apply(get_tolerance)
    merged['Pass'] = merged['AbsError'] <= merged['Tolerance']
    
    # Analyze by Checkpoint
    # チェックポイント別分析
    checkpoints = merged['CheckpointID'].unique()
    all_pass = True
    
    print("=" * 70)
    print("Per-Checkpoint Results / チェックポイント別結果")
    print("=" * 70)
    
    for cp in sorted(checkpoints):
        cp_data = merged[merged['CheckpointID'] == cp]
        cp_failures = cp_data[~cp_data['Pass']]
        tolerance = get_tolerance(cp)
        
        if cp_failures.empty:
            status = "✓ PASS"
            print(f"{cp:12s}: {status:10s} ({len(cp_data):6d} values, tolerance={tolerance:.0e})")
        else:
            status = "✗ FAIL"
            all_pass = False
            max_err = cp_data['AbsError'].max()
            print(f"{cp:12s}: {status:10s} ({len(cp_failures):6d}/{len(cp_data):6d} failures, "
                  f"tolerance={tolerance:.0e}, max_error={max_err:.6f})")
    
    print("=" * 70)
    
    # Overall result
    failures = merged[~merged['Pass']]
    
    if all_pass:
        print(f"\n[SUCCESS] All {len(merged)} values matched within per-checkpoint tolerances!")
        return True
    else:
        print(f"\n[FAIL] Found {len(failures)} total mismatches.")
        
        # Show first few failures for debugging
        print("\nFirst 10 Failures (for debugging):")
        print(failures.head(10)[['FrameID', 'CheckpointID', 'DimIndex', 'Value_GT', 'Value_Target', 'AbsError', 'Tolerance']].to_string())
        
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Default paths if not provided
        base_dir = "."
        gt_path = os.path.join(base_dir, "dump_openface.csv")
        target_path = os.path.join(base_dir, "dump_recover.csv")
    else:
        gt_path = sys.argv[1]
        target_path = sys.argv[2]

    success = verify_pipeline(gt_path, target_path)
    if not success:
        sys.exit(1)
