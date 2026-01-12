# debug_csv_source.py
# Check if FeatureExtraction's CSV output matches its dump
# FeatureExtractionのCSV出力がダンプと一致するか確認

import pandas as pd
import numpy as np

# Load the CSV that FeatureExtraction outputs
csv_path = 'samples/recover_au_test/test_bash1.csv'
csv_df = pd.read_csv(csv_path)

print("=" * 70)
print("CSV vs Dump Comparison")
print("=" * 70)

# Load dump_openface.csv (GT)
df_gt = pd.read_csv('dump_openface.csv', header=None, names=['FrameID','CP','Idx','Val'])
cp1_gt = df_gt[df_gt['CP']=='CP1'].sort_values(['FrameID', 'Idx'])

print(f"CSV file has {len(csv_df)} frames")
print(f"GT CP1 has FrameID range: {cp1_gt['FrameID'].min()} - {cp1_gt['FrameID'].max()}")

# Compare first 5 frames
print("\n--- Frame-by-Frame x_0 Comparison ---")
print("FrameID(GT) | CSV_x_0    | GT_x_0     | Match?")
print("-" * 50)

for gt_frame in range(5):
    # GT uses frames_tracking which for CP1 is BEFORE increment
    # So GT Frame 0 = input frame 0
    csv_row = gt_frame  # CSV frame index
    
    # Get CSV value
    csv_x0 = csv_df.iloc[csv_row]['x_0'] if csv_row < len(csv_df) else None
    
    # Get GT value
    gt_row = cp1_gt[(cp1_gt['FrameID']==gt_frame) & (cp1_gt['Idx']==0)]
    gt_x0 = gt_row['Val'].values[0] if len(gt_row) > 0 else None
    
    if csv_x0 is not None and gt_x0 is not None:
        match = "✓" if abs(csv_x0 - gt_x0) < 0.001 else "✗"
        err = abs(csv_x0 - gt_x0)
        print(f"{gt_frame:11} | {csv_x0:10.4f} | {gt_x0:10.4f} | {match} (err={err:.4f})")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)
print("""
If CSV_x_0 != GT_x_0, it means:
- FeatureExtraction dumps landmarks BEFORE they are saved to CSV
- OR the CSV was from a different/previous run
- OR there's some processing between dump and CSV save

Let's check the HOG file too - is it from the same run?
""")

# Check file modification times
import os
import datetime

hog_path = 'samples/recover_au_test/test_bash1.hog'
csv_mtime = os.path.getmtime(csv_path)
hog_mtime = os.path.getmtime(hog_path)
dump_gt_mtime = os.path.getmtime('dump_openface.csv')

print(f"test_bash1.csv modified: {datetime.datetime.fromtimestamp(csv_mtime)}")
print(f"test_bash1.hog modified: {datetime.datetime.fromtimestamp(hog_mtime)}")
print(f"dump_openface.csv modified: {datetime.datetime.fromtimestamp(dump_gt_mtime)}")

print("\nIf HOG and CSV have the same timestamp as dump_openface.csv,")
print("they are from the same FeatureExtraction run.")
