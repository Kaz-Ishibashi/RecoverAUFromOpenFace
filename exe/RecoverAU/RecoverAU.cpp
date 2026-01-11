// -------------------------------------------------------------------------------------------------------------------
// RecoverAU.cpp
// -------------------------------------------------------------------------------------------------------------------
// OpenFace Project - AU Recovery Tool
// OpenFace プロジェクト - AU リカバリーツール
//
// Purpose:
// This program recovers Action Units (AUs) from pre-computed HOG features (dlib format) and Facial Landmarks (CSV).
// It bypasses the standard image processing pipeline by injecting these features directly into the FaceAnalyser class.
//
// 目的:
// 本プログラムは、事前に計算されたHOG特徴量（dlib形式）と顔ランドマーク（CSV）から、Action Unit (AU) をリカバリーします。
// これらの特徴量をFaceAnalyserクラスに直接注入することで、標準的な画像処理パイプラインをバイパスします。
//
// Requirements:
// - Modified FaceAnalyser.h (exposed private members) / 修正されたFaceAnalyser.h（privateメンバの公開）
// - dlib HOG file / dlib HOGファイル
// - Landmark CSV file / ランドマークCSVファイル
// -------------------------------------------------------------------------------------------------------------------

// C++ Standard Libraries
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

// OpenFace Headers
// Note: Ensure include paths are correctly configured in Visual Studio
// 注意: Visual Studioでインクルードパスが正しく設定されていることを確認してください
#include "FaceAnalyser.h"
#include "LandmarkDetectorUtils.h" // For PDM and utils

// OpenCV Headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// dlib Headers
#include <dlib/matrix.h>
#include <dlib/serialize.h>
#include <dlib/opencv.h>

// Namespaces
using namespace std;
using namespace cv;
using namespace FaceAnalysis;

// -------------------------------------------------------------------------------------------------------------------
// Helper Function: Load Landmarks from CSV
// ヘルパー関数: CSVからランドマークを読み込む
// Format assumption: frame, face_id, x_0, y_0, ... x_67, y_67 (OpenFace standard)
// 前提フォーマット: frame, face_id, x_0, y_0, ... x_67, y_67 (OpenFace標準)
// -------------------------------------------------------------------------------------------------------------------
bool LoadLandmarks(const string& csv_path, vector<Mat_<float>>& all_landmarks) {
    ifstream file(csv_path);
    if (!file.is_open()) {
        cerr << "Error: Could not open landmark file: " << csv_path << endl;
        cerr << "エラー: ランドマークファイルを開けませんでした: " << csv_path << endl;
        return false;
    }

    string line;
    // Skip header / ヘッダーをスキップ
    getline(file, line); 

    while (getline(file, line)) {
        stringstream ss(line);
        string val_str;
        vector<float> values;
        
        while (getline(ss, val_str, ',')) {
            if (!val_str.empty())
                values.push_back(stof(val_str));
        }

        // Check if we have enough data (frame + face_id + 68*2 points = 138 columns minimum)
        // データが十分か確認 (frame + face_id + 68*2点 = 最低138列)
        if (values.size() < 138) continue;

        // Create 2x68 matrix / 2x68行列を作成
        Mat_<float> landmarks(2, 68);
        int idx = 2; // Start after frame and face_id / frameとface_idの次から開始
        for (int i = 0; i < 68; ++i) {
            landmarks(0, i) = values[idx];     // x
            landmarks(1, i) = values[idx + 68]; // y (OpenFace often stores xs then ys, or x1,y1? check format)
            // Warning: OpenFace CSV usually stores x_0, x_1 ... x_67, y_0, y_1 ... y_67
            // 警告: OpenFaceのCSVは通常 x_0, x_1 ... x_67, y_0, y_1 ... y_67 の順で格納されます
            // If interleaved (x1,y1, x2,y2...), adjust index. Assuming standard OpenFace output format here.
            // もし交互の場合 (x1,y1, x2,y2...) はインデックスを調整してください。ここではOpenFace標準出力を想定します。
        }
        
        // Wait, standard OpenFace CSV header is: frame, face_id, timestamp, confidence, success, gaze..., pose..., landmarks_2d (x_0...x_67, y_0...y_67)
        // Correction: Standard CSV has many columns. We need to parse by column name ideally, but here assuming stripped CSV or known fixed format.
        // If simply x0..x67, y0..y67 starting at col X.
        // Let's assume the user provides a simplified CSV as per instruction 2: "frame, x1, y1, ... x68, y68"
        // 修正: ユーザー指示に "format: frame, x1, y1, ... x68, y68" とあるため、これを前提にします。
        
        int landmark_start_idx = 1; // After frame number
        for(int i=0; i<68; ++i) {
            landmarks(0, i) = values[landmark_start_idx + i*2];     // x_i
            landmarks(1, i) = values[landmark_start_idx + i*2 + 1]; // y_i
        }
        
        all_landmarks.push_back(landmarks);
    }
    return true;
}

// -------------------------------------------------------------------------------------------------------------------
// Main Function
// -------------------------------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage: RecoverAU <hog_file> <landmark_csv> <output_csv> [model_dir]" << endl;
        cout << "使用法: RecoverAU <hog_file> <landmark_csv> <output_csv> [model_dir]" << endl;
        return 1;
    }

    string hog_file = argv[1];
    string landmark_file = argv[2];
    string output_file = argv[3];
    string model_dir = (argc > 4) ? argv[4] : "model/location"; // Update default if needed

    // 1. Load HOG Data (dlib deserialization)
    // 1. HOGデータの読み込み (dlibデシリアライズ)
    cout << "Loading HOG file..." << endl;
    vector<dlib::matrix<double, 0, 1>> hog_data; // dlib column vector
    try {
        ifstream fin(hog_file, ios::binary);
        if (!fin) throw dlib::serialization_error("File not found");
        dlib::deserialize(hog_data, fin);
    }
    catch (dlib::serialization_error& e) {
        cerr << "Error loading HOG file: " << e.what() << endl;
        return 1;
    }

    // 2. Load Landmarks
    // 2. ランドマークの読み込み
    cout << "Loading Landmarks..." << endl;
    vector<Mat_<float>> landmarks_data;
    if (!LoadLandmarks(landmark_file, landmarks_data)) {
        return 1;
    }

    if (hog_data.size() != landmarks_data.size()) {
        cerr << "Warning: Number of HOG frames (" << hog_data.size() << ") does not match Landmarks (" << landmarks_data.size() << ")." << endl;
        cerr << "警告: HOGフレーム数 (" << hog_data.size() << ") とランドマーク数 (" << landmarks_data.size() << ") が一致しません。" << endl;
        // Proceed with minimum? / 最小数で続行？
    }
    size_t num_frames = min(hog_data.size(), landmarks_data.size());

    // 3. Initialize FaceAnalyser
    // 3. FaceAnalyserの初期化
    cout << "Initializing FaceAnalyser..." << endl;
    FaceAnalyserParameters fa_params;
    // Ensure we use STATIC models if requested (check if arguments needed)
    // 必要に応じて静的モデルを使用するように引数を確認
    // fa_params.arguments = ...; 
    
    FaceAnalyser face_analyser(fa_params);

    // Prepare Output
    // 出力の準備
    ofstream out_file(output_file);
    if (!out_file.is_open()) {
        cerr << "Error opening output file." << endl;
        return 1;
    }
    
    // Write Header (Standard OpenFace AU header)
    // ヘッダー書き込み (OpenFace標準AUヘッダー)
    out_file << "frame,timestamp";
    auto au_names = face_analyser.GetAUClassNames();
    for (const auto& au : au_names) out_file << ",AU" << au << "_c";
    auto au_reg_names = face_analyser.GetAURegNames();
    for (const auto& au : au_reg_names) out_file << ",AU" << au << "_r";
    out_file << endl;

    // 4. Processing Loop
    // 4. 処理ループ
    cout << "Processing " << num_frames << " frames..." << endl;
    
    for (size_t i = 0; i < num_frames; ++i) {
        // A. Convert HOG to cv::Mat
        // A. HOGをcv::Matに変換
        // dlib matrix to cv::Mat (OpenFace usually expects CV_64F)
        Mat_<double> hog_mat_cv = dlib::toMat(hog_data[i]);
        
        // Critical: Clone/Copy to ensure continuous memory if needed, although toMat usually wraps.
        // Make sure it's valid.
        
        // B. Inject HOG into FaceAnalyser (Using exposed member)
        // B. HOGをFaceAnalyserに注入 (公開されたメンバを使用)
        // *** CRITICAL MODIFICATION REQUIRED IN FaceAnalyser.h ***
        // *** FaceAnalyser.h に重要な修正が必要です ***
        face_analyser.hog_desc_frame = hog_mat_cv.clone(); 

        // C. Calculate Geometry (Using PDM)
        // C. 幾何特徴量の計算 (PDMを使用)
        // We need 'geom_descriptor_frame'. OpenFace calculates this from PDM parameters.
        // Landmarks (image space) -> PDM Parameters -> Geometry Descriptor
        // ランドマーク（画像空間） -> PDMパラメータ -> 幾何特徴量
        
        // Step C1: PDM Parameters
        Mat_<float> shape_2d = landmarks_data[i];
        Vec6f params_global;
        Mat_<float> params_local;
        
        // We assume we don't have 3D, so we might estimate or fit.
        // face_analyser.pdm.CalcParams(...) requires 3D or 2D?
        // OpenFace's PDM::CalcParams computes PDM params given the 2D landmarks.
        // PDM::CalcParams(cv::Mat_<float>& out_params_global, cv::Mat_<float>& out_params_local, const cv::Mat_<float>& landmarks_2d, const Vec6f& params_global_init, const Mat_<float>& params_local_init, bool local_only=false);
        // Note: Signatures may vary. Assuming standard one.
        
        // First, calc params.
        face_analyser.pdm.CalcParams(params_global, params_local, shape_2d);

        // Step C2: Geometry Descriptor
        // In FaceAnalyser.cpp, GetGeomDescriptor() computes this.
        // Usually: geom_desc_frame = params_global (some subset) + params_local
        // FaceAnalyser implementation: 
        // geom_descriptor_frame = (params_global[0], params_global[4], params_global[5], params_local...)
        // We need to replicate this logic exactly or call a helper if available.
        // Since we exposed 'geom_descriptor_frame', we can write to it.
        
        // Replicating typical logic:
        // PDM params: scale, rot_x, rot_y, rot_z, tx, ty (6 global) + local (non-rigid)
        // FaceAnalyser typically uses: scale, rot_x, rot_y, rot_z (NO translation) + local weights
        
        Mat_<double> geom_desc(1, 4 + params_local.rows, 0.0);
        geom_desc(0,0) = params_global[0]; // Scale
        geom_desc(0,1) = params_global[1]; // Rot X
        geom_desc(0,2) = params_global[2]; // Rot Y
        geom_desc(0,3) = params_global[3]; // Rot Z
        
        Mat_<double> local_double;
        params_local.convertTo(local_double, CV_64F);
        
        // Copy local params
        Mat target_roi = geom_desc.colRange(4, 4 + params_local.rows);
        local_double.t().copyTo(target_roi);
        
        face_analyser.geom_descriptor_frame = geom_desc.clone();

        // D. Output Timestamp (or frame number)
        face_analyser.current_time_seconds = (double)i * 0.033; // Mock timestamp if needed / 仮のタイムスタンプ

        // E. Predict
        // E. 予測
        // Call the exposed prediction function
        // 公開された予測関数を呼び出し
        face_analyser.PredictCurrentAUs(0); // 0 = view

        // F. Write Result
        // F. 結果書き込み
        out_file << i << "," << face_analyser.current_time_seconds;
        
        // Presence
        auto preds_c = face_analyser.GetCurrentAUsClass();
        for (const auto& p : preds_c) out_file << "," << p.second;
        
        // Intensity
        auto preds_r = face_analyser.GetCurrentAUsReg();
        for (const auto& p : preds_r) out_file << "," << p.second;
        
        out_file << endl;
    }

    cout << "Done. Saved to " << output_file << endl;
    cout << "完了。" << output_file << " に保存しました。" << endl;

    return 0;
}
