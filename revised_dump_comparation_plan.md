# Revised Implementation Plan

## Goal Description
Reverse engineer OpenFace AU pipeline to achieve >0.99 correlation.
Key focus: Precise reproduction of "Offline Calibration" and verifying exact data match via intermediate dumps.

## Step 1: Final Pseudo-code Logic
(No changes, approved)

## Step 2: Dump Strategy & Formatting (Critical Update)

### 2.1 Dump Format
To ensure precise comparison, the `dump_var` function must output to a CSV file with high precision.
- **Format**: `FrameID, CheckpointID, DimensionIndex, Value`
- **Precision**: `std::setprecision(15)` (Double precision is required)
- **Files**:
    - `dump_openface.csv` (From modified OpenFace)
    - `dump_recover.csv` (From target implementation)

### 2.2 Checkpoints (Updated)
We will define Checkpoints (CP) in both codebases.

| Checkpoint | Content                            | Critical Note                                                        |
| ---------- | ---------------------------------- | -------------------------------------------------------------------- |
| **CP1**    | Raw Landmarks (x, y)               | Ensure exact input match.                                            |
| **CP2**    | Aligned Face (Raw Pixel Sum)       | Instead of full image, dump sum/mean of pixels to save space.        |
| **CP3**    | HOG Descriptor                     | *Warning: Heavy data.* Dump first 5 frames only for full validation. |
| **CP4**    | Geometry Descriptor                | Check PDM params + residuals.                                        |
| **CP5**    | Median Vector                      | Verify Histogram approximation logic.                                |
| **CP6**    | Raw Prediction (Before correction) | Output of SVR/SVM.                                                   |
| **CP7**    | **Calculated Offset (Cutoff)**     | *New*: The scalar value subtracted during offline phase.             |
| **CP8**    | **Final Value (After Smoothing)**  | *New*: The final output value.                                       |

## Step 3: Automated Verification (New)
Implement a Python script (`verify_pipeline.py`) to automatically compare `dump_openface.csv` and `dump_recover.csv`.

**Logic:**
1. Load both CSVs (pandas).
2. Group by CheckpointID.
3. Calculate absolute error: `|Val_OpenFace - Val_Recover|`.
4. **Pass Criteria**: Error < `1e-6` for all checkpoints.
5. Report the first FrameID/Index where divergence occurs.