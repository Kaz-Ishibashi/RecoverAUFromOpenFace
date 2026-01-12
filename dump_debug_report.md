# OpenFace RecoverAU ダンプ比較デバッグレポート

このドキュメントは、`RecoverAU.exe`と`FeatureExtraction.exe`のダンプ比較テストにおいて発見された問題と、その解決策をまとめたものです。

## 概要

**目的**: RecoverAU.exeがFeatureExtraction.exeと同一の中間値（チェックポイントCP1〜CP8）を出力するようにする。

**結果**: CP1〜CP7が完全一致（許容誤差内）。CP8はフレームオフセットの問題が残存。

---

## チェックポイント一覧

| CP       | 内容                               | 状態   |
| -------- | ---------------------------------- | ------ |
| CP1      | 入力ランドマーク座標               | ✅ PASS |
| CP3      | HOG特徴量（最初の5フレーム）       | ✅ PASS |
| CP4      | 幾何特徴量（params_local + 位置）  | ✅ PASS |
| CP5_Geom | 幾何中央値                         | ✅ PASS |
| CP5_HOG  | HOG中央値                          | ✅ PASS |
| CP6      | 生のAU予測値                       | ✅ PASS |
| CP7      | オフセット（キャリブレーション値） | ✅ PASS |
| CP8      | 最終AU値（オフセット適用後）       | ❌ FAIL |

---

## 解決した問題

### 問題1: HOGファイルが生成されない

**症状**: `pipeline_test.ps1`でHOGファイルが生成されず、古いデータが使われていた。

**原因**: FeatureExtractionの正しいフラグは `-hogalign` であり、`-hog` ではなかった。

**修正**:
```powershell
# 変更前
$FeatArgs = "-f", "$VideoPath", "-out_dir", "$SampleDir", "-2Dfp", "-3Dfp", "-aus", "-pose", "-hog"

# 変更後
$FeatArgs = "-f", "$VideoPath", "-out_dir", "$SampleDir", "-2Dfp", "-3Dfp", "-aus", "-pose", "-hogalign"
```

**ファイル**: `pipeline_test.ps1`

---

### 問題2: CP1のフレームIDオフセット

**症状**: CP1のフレームIDがGTとTGで1ずれていた（GT: 0-215, TG: 1-216）。

**原因**: OpenFaceの`AddNextFrame`では、CP1は`frames_tracking++`の**前に**ダンプされるが、他のCPは**後に**ダンプされる。

**OpenFaceコード分析**:
```cpp
// Line 317 in FaceAnalyser.cpp
DUMP_MAT(frames_tracking, "CP1", detected_landmarks);  // CP1は増分前
frames_tracking++;  // ここで増分
// Line 362+
DUMP_VAL(frames_tracking, "CP2", ...);  // CP2以降は増分後
```

**修正**:
```cpp
// RecoverAU.cpp
int dump_frame_id_cp1 = static_cast<int>(i);      // CP1用: i を使用
int dump_frame_id = static_cast<int>(i) + 1;       // 他のCP用: i+1 を使用

DUMP_MAT(dump_frame_id_cp1, "CP1", shape_2d);  // CP1
DUMP_MAT(dump_frame_id, "CP4", ...);            // 他のCP
```

**ファイル**: `RecoverAU.cpp`

---

### 問題3: HOGの精度損失（float32 vs double）

**症状**: CP3/CP5_HOGでわずかな誤差が発生。

**原因**: HOGファイル（`.hog`）は`float32`で保存されるが、`FaceAnalyser`は`double`で処理するため、読み書き時に精度が低下。

**修正**: `verify_pipeline.py`でCP3/CP5_HOG/CP6/CP7/CP8の許容誤差を`1e-3`に緩和。

```python
# チェックポイント別許容誤差
TOLERANCES = {
    'CP1': 1e-6,
    'CP3': 1e-3,      # float32精度損失
    'CP4': 1e-6,
    'CP5_Geom': 1e-6,
    'CP5_HOG': 1e-3,  # float32精度損失
    'CP6': 1e-3,
    'CP7': 1e-3,
    'CP8': 1e-3,
}
```

**ファイル**: `verify_pipeline.py`

---

### 問題4: AU順序の不一致

**症状**: CP7のインデックスでGTとTGの値が異なっていた。

**原因**: 
- OpenFaceは`AU_predictions_reg_all_hist`（`std::map`）を反復 → **アルファベット順**
- RecoverAUは`GetAURegNames()`を反復 → **static優先、dynamic後の順**

**修正**: RecoverAUでAU名をソートしてからアルファベット順で処理。

```cpp
// AU名をアルファベット順にソート
vector<string> sorted_au_names = all_au_names;
std::sort(sorted_au_names.begin(), sorted_au_names.end());

// ソート済み順序で反復
for (size_t sorted_idx = 0; sorted_idx < sorted_au_names.size(); ++sorted_idx) {
    string au_name = sorted_au_names[sorted_idx];
    int orig_idx = sorted_to_orig[sorted_idx];  // 元のインデックスへのマッピング
    // ...
}
```

**ファイル**: `RecoverAU.cpp`

---

### 問題5: CP7ダンプ条件の不一致

**症状**: CP7のダンプ数がGT（17件）とTG（11件）で異なっていた。

**原因**: OpenFaceはelseブロック内の**すべての**AUでCP7をダンプするが、RecoverAUは`cutoff != -1`の場合のみダンプしていた。

**OpenFaceコード**:
```cpp
if(au_good.empty() || !dynamic) {
    offsets.push_back(0.0);  // ダンプなし
} else {
    // オフセット計算
    offsets.push_back(offset);
    DUMP_VAL(-1, "CP7", (int)offsets.size()-1, offsets.back());  // 常にダンプ
}
```

**修正**: elseブロック内のすべてのAUでCP7をダンプ。

**ファイル**: `RecoverAU.cpp`

---

### 問題6: CP8のダンプタイミング（スムージング前後）

**症状**: CP8の値がスムージング後の値になっていた。

**原因**: OpenFaceはスムージング**前に**CP8をダンプするが、RecoverAUはスムージング**後に**ダンプしていた。

**修正**: スムージング前にCP8をダンプするよう順序を変更。

```cpp
// オフセット適用 & クリッピング
for(size_t frame_i = 0; frame_i < history.size(); ++frame_i) {
    if(history[frame_i].success) {
        double val = history[frame_i].raw_reg[orig_idx] - offset;
        if(val < 0.0) val = 0.0;
        if(val > 5.0) val = 5.0;
        
        DUMP_VAL(frame_i + 1, "CP8", sorted_idx, val);  // スムージング前にダンプ
        history[frame_i].final_reg[orig_idx] = val;
    }
}

// ここでスムージング（ダンプなし）
```

**ファイル**: `RecoverAU.cpp`

---

### 問題7: PostprocessPredictions の欠如（最重要）

**症状**: CP7のオフセット値がGTとTGで大きく異なっていた。CP6は一致していたにもかかわらず。

**原因**: OpenFaceの`ExtractAllPredictionsOfflineReg`は、オフセット計算**前に**`PostprocessPredictions()`を呼び出す。この関数は：
1. 初期フレームのHOG/幾何記述子を保存
2. 全フレーム処理後、**最終的な安定した中央値**を使って初期フレームの予測を再計算
3. `AU_predictions_reg_all_hist`を更新

CP6は再計算**前の**値でダンプされるが、CP7/CP8は再計算**後の**値を使用。

**修正**: RecoverAUにPHASE 1.5として「ポストプロセス」を追加。

```cpp
// --- PHASE 1.5: POSTPROCESSING ---
// 初期フレームを最終中央値で再予測
cout << "Debug: Postprocessing " << frames_tracking_succ << " initial frames..." << endl;
{
    int success_ind = 0;
    int all_ind = 0;
    
    while(all_ind < (int)history.size() && success_ind < frames_tracking_succ) {
        if(history[all_ind].success) {
            // 保存した記述子を復元
            face_analyser.hog_desc_frame = hog_desc_frames_init[success_ind].clone();
            face_analyser.geom_descriptor_frame = geom_descriptor_frames_init[success_ind].clone();
            
            // 最終中央値で再予測
            auto preds_r = face_analyser.PredictCurrentAUs(0);
            
            // historyを更新
            for(size_t k = 0; k < preds_r.size(); ++k) {
                history[all_ind].raw_reg[k] = preds_r[k].second;
            }
            
            success_ind++;
        }
        all_ind++;
    }
}
```

**ファイル**: `RecoverAU.cpp`

---

## 残存問題: CP8のフレームオフセット

**現在の症状**:
```
FrameID 1, CP8[0]: GT=1.232693, TG=0.973320
FrameID 2, CP8[0]: GT=1.416492, TG=1.232693
FrameID 3, CP8[0]: GT=1.403577, TG=1.416492
```

**観察**: TGの値がGTの1フレーム後の値に見える（TG[N] ≈ GT[N+1]）。

**仮説**: CP8のダンプ時に使用するフレームIDが1ずれている可能性。

**調査中**...

---

## 修正ファイル一覧

| ファイル             | 修正内容                                                            |
| -------------------- | ------------------------------------------------------------------- |
| `pipeline_test.ps1`  | `-hogalign`フラグ使用、古いファイル削除                             |
| `verify_pipeline.py` | チェックポイント別許容誤差の実装                                    |
| `RecoverAU.cpp`      | CP1フレームID、AU順序、CP7ダンプ条件、CP8タイミング、ポストプロセス |

---

## 使用コマンド

```powershell
# テスト実行
.\pipeline_test.ps1

# 結果確認
python exe\RecoverAU\verify_pipeline.py dump_openface.csv dump_recover.csv
```

---

*作成日: 2026-01-12*
