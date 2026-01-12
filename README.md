# OpenFace AU Recovery Tool / OpenFace AU 復元ツール

**[English]**
This project provides a solution for recovering **Action Unit (AU)** predictions from OpenFace output files (HOG features and Landmarks) when the AU calculation was accidentally omitted (e.g., forgetting the `-aus` flag). Instead of re-processing the entire video, which can be computationally expensive or impossible if the source video is lost, this tool (`RecoverAU`) strictly replicates OpenFace's internal AU prediction, post-processing, and smoothing logic to generate identical results.

**[日本語]**
このプロジェクトは、OpenFaceの出力ファイル（HOG特徴量とランドマーク）から、**Action Unit (AU)** の予測値を復元するためのソリューションです。`-aus` フラグを付け忘れてAUが出力されなかった場合などに有用です。動画全体を再処理するのではなく（計算コストが高く、元の動画がない場合は不可能）、HOGとランドマークからOpenFace内部のAU予測、ポストプロセス、スムージングロジックを厳密に再現することで、元のOpenFaceと完全に一致する結果を生成します。

---

## 🎯 Motivation / 目的

Have you ever run a long OpenFace batch process only to realize you forgot the `-aus` flag?
OpenFace outputs a `.hog` file (binary HOG features) and a `.csv` (Landmarks) if configured. This project utilizes these artifacts to calculate AUs **offline**, saving time and resources.

OpenFaceの長時間バッチ処理を実行した後で、`-aus` フラグを付け忘れたことに気づいたことはありませんか？
OpenFaceは設定により `.hog` ファイル（バイナリHOG特徴量）と `.csv`（ランドマーク）を出力します。本プロジェクトは、これらのアーティファクトを利用して **オフラインで** AUを計算し、時間とリソースを節約します。

## 🛠️ Technology Stack & Customization / 技術スタックとカスタマイズ

### Core Components

- **Base Framework**: [OpenFace 2.2.0](https://github.com/TadasBaltrusaitis/OpenFace) (C++)
- **New Executable**: `RecoverAU.exe`
  - A standalone C++ application linked against OpenFace libraries (`FaceAnalyser`, `LandmarkDetector`).
  - Re-implements the prediction pipeline found in `FeatureExtraction` but decouples it from video input.
- **Verification Scripts**: Python (`pandas`, `numpy`) & PowerShell.

### Key Technical Challenges & Solutions / 主な技術的課題と解決策

1. **Exact Reproduction of Logic / ロジックの完全再現**:
    OpenFace's AU prediction involves more than just passing HOG features to an SVM/SVR. It requires specific **Post-processing** and **Smoothing**:
    - **Calibration**: Calculating offsets based on the lowest n-percentile of predictions (to handle individual face resting neutral expression).
    - **Smoothing**: Moving average filtering (Window size 7 for Classification, 3 for Regression).
    - **Data Alignment**: Handling discrepancies in HOG usage (row-major vs column-major) and frame indexing.

2. **Reverse Engineering FaceAnalyser / FaceAnalyserのリバースエンジニアリング**:
    We deeply analyzed `FaceAnalyser.cpp` to understand how it buffers "initial frames" to calibrate the neutral expression and how it applies dynamic AU correctors.

## 🚀 Development Methodology: "Checkpoint Verification" / 開発手法：チェックポイント検証

To ensure the recovered AUs match the "Ground Truth" (what OpenFace *would* have produced) with **>99% correlation**, we developed a rigorous debugging methodology called **Checkpoint Verification**.

**Accuracy Achieved**: Matches OpenFace output with a correlation of **1.000** (Regression) and F1-Score **1.000** (Classification).

### Checkpoints (CP)

We instrumented both the official OpenFace code and our `RecoverAU` code to dump intermediate values at critical steps:

| CP ID     | Description             | Purpose                                                                |
| :-------- | :---------------------- | :--------------------------------------------------------------------- |
| **CP1**   | Raw Landmarks           | Verify input CSV parsing and coordinate systems.                       |
| **CP3-5** | HOG Features            | Verify binary HOG loading and normalization (Row/Col major fixes).     |
| **CP6**   | Raw Regression (SVR)    | Verify raw model predictions before calibration.                       |
| **CP7**   | Calibration Offsets     | Verify specific per-AU offsets calculated from the prediction history. |
| **CP9**   | Head Pose (Rigid)       | Verify PDM (Point Distribution Model) parameter fitting.               |
| **CP11**  | Raw Classification      | Verify raw SVM outputs.                                                |
| **CP12**  | Smoothed Classification | Verify moving average and thresholding logic.                          |

By comparing these checkpoints frame-by-frame (tolerance `1e-6`), we pinpointed and fixed subtle bugs (e.g., sorting order of AU names in maps vs vectors).

復元されたAUが「Ground Truth」（OpenFaceが出力するはずだった値）と **相関係数 0.99以上** で一致することを保証するために、**チェックポイント検証 (Checkpoint Verification)** と呼ぶ厳密なデバッグ手法で開発しました。
最終的に、回帰（Regression）で **相関 1.000**、分類（Classification）で **F1スコア 1.000** を達成しました。

公式OpenFaceと `RecoverAU` の両方のコードに、処理の重要段階で中間値を出力する「チェックポイント」を埋め込みました（上記表参照）。これらをフレーム単位で比較（許容誤差 `1e-6`）することで、AU名のソート順序の違いや、HOGの行列構造の違いなど、微細なバグを特定し修正しました。

## 📦 Usage / 使用方法

### Prerequisite / 前提条件

- You must have the `.hog` file generated by OpenFace (requires `-hogalign` flag during original extraction).
- You must have the `.csv` file with landmarks.
- OpenFace models must be present in the execution directory.

### Build

1. Open `OpenFace.sln` in Visual Studio.
2. Build the `RecoverAU` project (Release / x64).

### Run

```powershell
# Usage: RecoverAU.exe -f <hog_file> -l <landmark_csv> -out_dir <output_dir>
.\x64\Release\RecoverAU.exe -f "video.hog" -l "video.csv" -out_dir "output/"
```

### Verification (Optional)

Use the included `verify_recovery.py` to compare your recovered CSV against a ground truth CSV (if available) to ensure accuracy.

```bash
python verify_recovery.py compare --gt ground_truth.csv --rec recovered.csv
```

## 📝 License / ライセンス

This project is an extension of [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) and adheres to its license terms.
