# OpenFace & RecoverAU logic breakdown / OpenFaceとRecoverAUのロジック詳細解説

本ドキュメントでは、`FeatureExtraction`（オリジナル）が動画からAction Units (AU) を算出するロジックと、`RecoverAU`（リカバリツール）が外部ファイル（HOG/Landmarks）を用いてその処理を再現する仕組み、および中間値比較用チェックポイント（CP1〜CP8）の配置について解説します。

---

## 1. FeatureExtraction のAU算出ロジック

`FeatureExtraction` は、動画フレームを入力とし、顔追跡（Landmark Detection）、特徴量抽出、AU予測を行うパイプラインです。

### 処理フロー概要
1. **画像取得**: 動画からフレームを取得
2. **顔追跡 (Landmark Detection)**: `LandmarkDetector::DetectLandmarksInVideo`
   - ここで68点のランドマーク（2D/3D）が特定される。
3. **FaceAnalyser::AddNextFrame**: 顔分析のメイン処理
   - **CP1 (Raw Landmarks)**: 検出されたランドマークを受け取る。
   - **幾何特徴量計算 (Geometric Descriptors)**: ランドマークから形状パラメータ（params_local）や相対位置を計算。
     - **CP4 (Geom Desc)**: 計算された幾何特徴量。
     - **CP5_Geom (Median)**: 幾何特徴量の移動中央値（ノイズ除去用）。
   - **HOG特徴量抽出 (HOG Descriptors)**: ランドマーク位置に基づいてHOG特徴量を抽出。
     - **CP3 (HOG Raw)**: 抽出された生のHOG特徴量。
     - **CP5_HOG (Median)**: HOG特徴量の移動中央値。
   - **AU予測 (Prediction)**: SVM/SVRを用いてAU強度・有無を予測。
     - 「幾何特徴量」と「HOG特徴量」の両方を入力とする（AUによって異なる）。
     - **CP6 (Raw Prediction)**: 直後の予測値。
4. **履歴保存**: 予測値を履歴（History）に保存。
5. **オフライン・キャリブレーション (Post-processing)**: 全フレーム処理後、個人の癖（バイアス）を補正。
   - `FaceAnalyser::ExtractAllPredictionsOfflineReg`
   - **PostprocessPredictions**: **初期フレーム（デフォルト3000フレームまで）を「最終的な中央値」を使って再予測**する。
     - *重要*: CP6の時点では初期の中央値を使っているが、最終出力ではこの再予測値が使われる。
   - **CP7 (Offfset)**: ダイナミックAU（動きで定義されるAU）のベースライン（Offset）を計算。
   - **CP8 (Final Value)**: オフセットを引き、クリッピング（0-5）した最終値。

---

## 2. RecoverAU の再現ロジック

`RecoverAU` は、動画処理を行わず、保存されたHOGファイルとランドマークCSVから、上記 `FaceAnalyser` の内部状態を「偽装」して再現します。

### データの割り込みポイント (Injection Points)

通常の `LandmarkDetector` をバイパスし、`FaceAnalyser` のメンバ変数を直接書き換えることで処理を再現しています。

| データ           | 元のソース           | RecoverAUでの注入方法                        |
| ---------------- | -------------------- | -------------------------------------------- |
| **Landmarks**    | 動画から検出         | CSVファイル (`LoadLandmarks`) から読み込み   |
| **HOG**          | 画像から抽出         | `.hog`ファイル (`ReadHOGFrame`) から読み込み |
| **Success Flag** | トラッキング成功判定 | CSVの `success` カラム読み込み               |

### 処理フローとチェックポイントの配置

RecoverAUのループ処理（`RecoverAU.cpp`）における各ステップとCPの対応です。

#### Phase 1: オンライン処理の再現 (Frame Loop)

フレームごとに以下を実行します。

1. **データ注入**:
   - `face_analyser.hog_desc_frame` ← HOGデータ
   - `face_analyser.geom_descriptor_frame` ← ランドマークから計算
   - **CP1**: ランドマーク座標をダンプ。
     - *補足*: `frames_tracking`インクリメント前のタイミングで行う（GTと一致させるため）。

2. **特徴量計算**:
   - `hog_desc_frame` の転置処理（行/列ベクトルの整合性確保）。
   - **CP3**: HOG特徴量をダンプ。
   - **CP4**: 幾何特徴量をダンプ。

3. **中央値更新 (Running Median)**:
   - CSVの `success` フラグに基づき、HOG/幾何特徴量の中央値を更新。
   - **CP5_Geom / CP5_HOG**: 更新後の中央値をダンプ。

4. **AU予測 (PredictCurrentAUs)**:
   - 現在の特徴量と中央値を使って予測を実行。
   - **CP6**: 予測された生のAU値（`raw_reg`）をダンプ。

5. **初期データ保存**:
   - ポストプロセスのために、初期フレームのHOG/幾何特徴量をメモリに保存（`hog_desc_frames_init`）。

#### Phase 1.5: ポストプロセス (今回の修正で追加)

OpenFaceの `PostprocessPredictions` を再現するフェーズです。

- 全フレーム処理後、保存しておいた初期フレームの特徴量をロード。
- **最終的に安定した中央値** を使って `PredictCurrentAUs` を再実行。
- 履歴（History）の予測値を、この再計算値で上書き。
- *理由*: CP6は「その時点の中央値」での予測ですが、CP7以降の計算には「最終中央値」での予測結果が使われるため、ここを合わせないとCP7/CP8が一致しません。

#### Phase 2: オフライン・キャリブレーション (Offline Calibration)

OpenFaceの `ExtractAllPredictionsOfflineReg` を再現します。

1. **AU順序の整列**:
   - OpenFaceは `std::map` で処理するため、AUは **アルファベット順** に処理されます。
   - RecoverAUもこれに合わせてAU名をソート。

2. **オフセット計算**:
   - 各AUについて、予測値のヒストグラムから下位X%（Cutoff）の値をオフセットとして計算。
   - **CP7**: 計算されたオフセット値をダンプ。
     - *条件*: ダイナミックAUでなくても（値0でも）、else分岐に入るすべてのAUについてダンプする（OpenFaceの挙動に一致）。

3. **最終値計算**:
   - 全フレームに対し、`予測値 - オフセット` を計算。
   - 0〜5の範囲にクリッピング。
   - **CP8**: スムージング **前** の値をダンプ。
     - *補足*: フレームIDは0-indexedでダンプ（OpenFace変数 `frame` と一致）。

4. **スムージング**:
   - 移動平均（Window=3）を適用し、最終出力（CSV）へ。

---

## 3. チェックポイント (CP) 詳細一覧

| CP Output ID | 内容            | FeatureExtraction (C++) の場所 | RecoverAU (C++) の場所 | 備考                                  |
| ------------ | --------------- | ------------------------------ | ---------------------- | ------------------------------------- |
| **CP1**      | Raw Landmarks   | `FaceAnalyser.cpp` L317        | `RecoverAU.cpp` L402   | 唯一 `frames_tracking` 増分前にダンプ |
| **CP3**      | HOG Descriptor  | `FaceAnalyser.cpp` L419        | `RecoverAU.cpp` L420   | 最初の5フレームのみダンプ             |
| **CP4**      | Geom Descriptor | `FaceAnalyser.cpp` L380        | `RecoverAU.cpp` L438   | 剛体パラメータなど                    |
| **CP5_Geom** | Geom Median     | `FaceAnalyser.cpp` L489        | `RecoverAU.cpp` L477   |                                       |
| **CP5_HOG**  | HOG Median      | `FaceAnalyser.cpp` L496        | `RecoverAU.cpp` L485   |                                       |
| **CP6**      | Raw Prediction  | `FaceAnalyser.cpp` L513        | `RecoverAU.cpp` L548   | 再予測前の値                          |
| **CP7**      | Offset          | `FaceAnalyser.cpp` L643        | `RecoverAU.cpp` L646   | ダイナミックAU補正値                  |
| **CP8**      | Final Reg Value | `FaceAnalyser.cpp` L668        | `RecoverAU.cpp` L753   | オフセット適用後、スムージング前      |

---

このロジックにより、RecoverAUは入力ソース（動画 vs ファイル）の違いを超えて、OpenFaceとビット単位でほぼ一致する計算結果を再現可能としています。
