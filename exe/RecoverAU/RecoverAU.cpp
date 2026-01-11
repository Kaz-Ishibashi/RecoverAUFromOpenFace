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
// -------------------------------------------------------------------------------------------------------------------

// C++ Standard Libraries
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm> // For min

// OpenFace Headers
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
// HOG Reading Helper
// -------------------------------------------------------------------------------------------------------------------
// RecorderHOG Format:
// int num_cols (4 bytes)
// int num_rows (4 bytes)
// int num_channels (4 bytes)
// float good_frame (4 bytes)
// float data[] (rows * cols * channels * 4 bytes) - Row Major? RecorderHOG iterates y(cols), x(rows), o(31) which suggests Col-Major logic but it writes linearly.
// Let's assume standard looping: y(0..cols), x(0..rows), o(0..31)

struct RawHOGFrame {
    int num_cols;
    int num_rows;
    int num_channels;
    bool good_frame;
    cv::Mat_<double> hog_data; // Converted to double for usage
};

bool ReadHOGFrame(ifstream& fin, RawHOGFrame& frame) {
    if (!fin.good() || fin.peek() == EOF) return false;

    // Read Dimensions
    fin.read((char*)&frame.num_cols, 4);
    if (!fin) return false;
    fin.read((char*)&frame.num_rows, 4);
    fin.read((char*)&frame.num_channels, 4);

    // Read Good Frame flag (stored as float)
    float good_frame_float;
    fin.read((char*)&good_frame_float, 4);
    frame.good_frame = (good_frame_float > 0);

    // Read Data
    int total_elements = frame.num_cols * frame.num_rows * frame.num_channels;
    if (total_elements <= 0) return false;

    // Data is stored as float in the file
    vector<float> raw_data(total_elements);
    // Assume file is written in standard Raster Order (Row-Major):
    // Outer: Rows (x=0..num_rows-1)
    // Inner: Cols (y=0..num_cols-1)
    // Deepest: Channels
    // Index = row * num_cols * chans + col * chans + ch
    fin.read((char*)raw_data.data(), total_elements * 4);
    
    // Check read success
    if (!fin) return false;

    // Convert to Mat_<double> and Permute to OpenFace Layout
    // OpenFace (Face_utils.cpp: Extract_FHOG_descriptor) flattens as:
    // Outer: Cols (y=0..num_cols-1)
    // Inner: Rows (x=0..num_rows-1)
    // Deepest: Channels
    // Target Index = col * num_rows * chans + row * chans + ch
    
    frame.hog_data.create(1, total_elements);
    
    int num_rows = frame.num_rows;
    int num_cols = frame.num_cols;
    int num_channels = frame.num_channels;

    for (int y = 0; y < num_cols; ++y) { // Col matches OpenFace outer
        for (int x = 0; x < num_rows; ++x) { // Row matches OpenFace inner
            for (int ch = 0; ch < num_channels; ++ch) {
                
                // Destination Index (Col-Major linear)
                // Since we fill standard linear buffer: dest_idx++;
                int dest_idx = y * (num_rows * num_channels) + x * num_channels + ch;
                
                // Source Index (Row-Major assumption from file)
                // src_idx = x * (num_cols * num_channels) + y * num_channels + ch;
                int src_idx = x * (num_cols * num_channels) + y * num_channels + ch;
                
                if (dest_idx < total_elements && src_idx < total_elements) {
                    frame.hog_data(0, dest_idx) = (double)raw_data[src_idx];
                }
            }
        }
    }
    
    return true;
}


// -------------------------------------------------------------------------------------------------------------------
// Helper Function: Load Landmarks from CSV
// -------------------------------------------------------------------------------------------------------------------
bool LoadLandmarks(const string& csv_path, vector<Mat_<float>>& all_landmarks) {
    cout << "Debug: Opening landmark file: " << csv_path << endl;
    ifstream file(csv_path);
    if (!file.is_open()) {
        cerr << "Error: Could not open landmark file: " << csv_path << endl;
        return false;
    }

    string line;
    // Skip header
    if (!getline(file, line)) {
        cerr << "Error: Landmark file appears empty." << endl;
        return false;
    }
    cout << "Debug: Header skipped. Reading lines..." << endl;

    int row_count = 0;
    while (getline(file, line)) {
        stringstream ss(line);
        string val_str;
        vector<float> values;
        
        while (getline(ss, val_str, ',')) {
            if (!val_str.empty()) {
                try {
                    values.push_back(stof(val_str));
                } catch (...) {
                     // ignore parse errors for now
                }
            }
        }

        // Check if we have enough data (frame + face_id + 68*2 points = 138 columns minimum)
        if (values.size() < 138) {
            continue; 
        }

        Mat_<float> landmarks(2, 68);
        
        // Assumption: format is frame, x1, y1, ... x68, y68 (interleaved)
        int landmark_start_idx = 1; // After frame number
        
        for(int i=0; i<68; ++i) {
            landmarks(0, i) = values[landmark_start_idx + i*2];     // x_i
            landmarks(1, i) = values[landmark_start_idx + i*2 + 1]; // y_i
        }
        
        all_landmarks.push_back(landmarks);
        row_count++;
    }
    cout << "Debug: Loaded " << row_count << " landmark frames." << endl;
    return true;
}

// -------------------------------------------------------------------------------------------------------------------
// Main Function
// -------------------------------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
    cout << "Debug: Program started." << endl;

    if (argc < 4) {
        cout << "Usage: RecoverAU <hog_file> <landmark_csv> <output_csv> [model_dir]" << endl;
        return 1;
    }

    string hog_file = argv[1];
    string landmark_file = argv[2];
    string output_file = argv[3];

    // 1. Load HOG Data (Custom Format)
    cout << "Debug: Loading HOG file: " << hog_file << endl;
    vector<RawHOGFrame> hog_frames;
    
    try {
        ifstream fin(hog_file, ios::binary);
        if (!fin) {
             cerr << "Error: HOG file not found or cannot be opened." << endl;
             return 1;
        }

        cout << "Debug: Reading custom HOG frames..." << endl;
        while (true) {
            RawHOGFrame frame;
            if (ReadHOGFrame(fin, frame)) {
                hog_frames.push_back(frame);
            } else {
                break; // EOF or error
            }
        }
        
        cout << "Debug: Successfully loaded HOG data. Frames: " << hog_frames.size() << endl;
        if(hog_frames.size() > 0) {
            cout << "Debug: First frame info - Rows: " << hog_frames[0].num_rows 
                 << ", Cols: " << hog_frames[0].num_cols 
                 << ", Chans: " << hog_frames[0].num_channels << endl;
        }
    }
    catch (std::exception& e) {
        cerr << "Error: Exception during HOG loading: " << e.what() << endl;
        return 1;
    }

    // 2. Load Landmarks
    cout << "Debug: Loading Landmarks file: " << landmark_file << endl;
    vector<Mat_<float>> landmarks_data;
    if (!LoadLandmarks(landmark_file, landmarks_data)) {
        cerr << "Error: Failed to load landmarks." << endl;
        return 1;
    }

    // Validation
    if (hog_frames.empty()) {
        cerr << "Error: HOG data is empty." << endl;
        return 1;
    }
    if (landmarks_data.empty()) {
        cerr << "Error: Landmark data is empty." << endl;
        return 1;
    }

    if (hog_frames.size() != landmarks_data.size()) {
        cerr << "Warning: HOG frames (" << hog_frames.size() << ") != Landmarks (" << landmarks_data.size() << ")." << endl;
    }
    size_t num_frames = min(hog_frames.size(), landmarks_data.size());
    cout << "Debug: Processing " << num_frames << " frames." << endl;

    // 3. Initialize FaceAnalyser
    cout << "Debug: Initializing FaceAnalyser..." << endl;
    
    // Construct arguments to pass to FaceAnalyserParameters
    // We pass argv[0] so it can find the root directory.
    // Also explicitly tell it where to look if possible, but the constructor logic is:
    // root = fs::path(argv[0]).parent_path();
    // Then it looks for "AU_predictors/..." relative to root or current dir?
    // Actually FaceAnalyserParameters::init() sets model_location.
    // By default it might look in "./AU_predictors".
    // Since we are in OpenFace root (CWD), and models are in lib/local/FaceAnalyser/AU_predictors,
    // we need to guide it. 
    // BUT usually OpenFace "install" copies models to the bin folder.
    // If not, we might fail.
    // Let's rely on valid[0] = true logic or passed args.
    
    vector<string> fa_args;
    fa_args.push_back(string(argv[0]));
    // If user wants static, add "-au_static"? User requested static before.
    fa_args.push_back("-au_static"); 
    
    FaceAnalyserParameters fa_params(fa_args);

    // Manual override if needed: 
    // If we are strictly in the dev structure:
    // "model_location" is private. We can't set it.
    // But we can rely on file system operations or CWD.
    // The previous error was "Could not find...".
    // Let's assume standard behavior: copy generic assumption.
    
    try {
        FaceAnalyser face_analyser(fa_params);
        
        // Critical Check: Did it load?
        if (face_analyser.pdm.NumberOfPoints() == 0) {
            cerr << "Error: FaceAnalyser failed to initialize PDM. Models likely not found." << endl;
            cerr << "Hint: Ensure 'AU_predictors' directory is in the current directory or relative to executable." << endl;
            // Try to suggest where it is looking?
            // cerr << "Search path: " << fa_params.getModelLoc() << endl; // access if public
            return 1;
        }

        if (face_analyser.GetAUClassNames().empty() && face_analyser.GetAURegNames().empty()) {
             cerr << "Error: No AU models loaded." << endl;
             return 1;
        }

        cout << "Debug: FaceAnalyser initialized. PDM Points: " << face_analyser.pdm.NumberOfPoints() << endl;

        // Prepare Output
        cout << "Debug: Opening output file: " << output_file << endl;
        ofstream out_file(output_file);
        if (!out_file.is_open()) {
            cerr << "Error: Could not open output file." << endl;
            return 1;
        }
        
        // Write Header
        out_file << "frame,timestamp";
        auto au_names = face_analyser.GetAUClassNames();
        for (const auto& au : au_names) out_file << "," << au << "_c";
        auto au_reg_names = face_analyser.GetAURegNames();
        for (const auto& au : au_reg_names) out_file << "," << au << "_r";
        out_file << endl;

        // 4. Processing Loop
        cout << "Debug: Starting processing loop..." << endl;
        
        for (size_t i = 0; i < num_frames; ++i) {
            // if (i % 100 == 0) cout << "Debug: Processing frame " << i << "/" << num_frames << endl;

            // A. HOG
            // Set dimensions
            face_analyser.num_hog_rows = hog_frames[i].num_rows;
            face_analyser.num_hog_cols = hog_frames[i].num_cols;
            
            // HOG Data: Convert to standard Row Vector (1 x N)
            // FaceAnalyser expects Row Vectors for feature concatenation (hconcat)
            face_analyser.hog_desc_frame = hog_frames[i].hog_data.clone(); 
            
            // B. Calculate Geometry
            Mat_<float> shape_2d = landmarks_data[i];

            // ★修正ポイント1：HOGが「縦長」なら「横長」に転置する
            if (face_analyser.hog_desc_frame.rows > face_analyser.hog_desc_frame.cols) {
                face_analyser.hog_desc_frame = face_analyser.hog_desc_frame.t();
            }
            
            // Check shape dimensions: should be 68 rows x 2 cols?
            // LoadLandmarks creates 2 rows x 68 cols (x1..x68; y1..y68)
            // But OpenFace PDM usually works with COLUMN vectors (2*n x 1) or (n x 2) or (n x 3)?
            // PDM::CalcParams: const cv::Mat_<float> & landmark_locations
            // It expects a ONE-COLUMN matrix of size 2*n x 1 (x1...xn, y1...yn)^T?
            // Or (n x 2)?
            // Let's check PDM.cpp:
            // "landmark_locations.at<float>(i)" -> Accessing index i.
            // "landmark_locations.at<float>(i+n)" -> Accessing index i+n.
            // This implies it expects a single column vector (2n x 1) where first n are X, next n are Y.
            // My LoadLandmarks creates (2, 68). THIS IS THE TYPE ERROR!
            // I need to reshape/transpose it to (136, 1).
            
            Mat_<float> shape_2d_formatted(face_analyser.pdm.NumberOfPoints() * 2, 1);
            for(int k=0; k<face_analyser.pdm.NumberOfPoints(); ++k) {
                shape_2d_formatted(k, 0) = shape_2d(0, k); // x
                shape_2d_formatted(k + face_analyser.pdm.NumberOfPoints(), 0) = shape_2d(1, k); // y
            }
            
            Vec6f params_global;
            Mat_<float> params_local;
            
            face_analyser.pdm.CalcParams(params_global, params_local, shape_2d_formatted);
            
            // ★重要修正: OpenFaceのFaceAnalyser.cppと同じgeom_descriptor構造を使用
            // FaceAnalyserの正しい構造: [locs (princ_comp * local) | local] = [204 | 34] = 238次元
            // 旧コード(間違い): [pose(4) | local(34)] = 38次元 → means(4702)と不一致
            
            // params_localを転置して行ベクトルに
            Mat_<double> local_params_row;
            params_local.convertTo(local_params_row, CV_64F);
            local_params_row = local_params_row.t(); // 1 x 34 行ベクトル
            
            // princ_comp を double に変換
            Mat_<double> princ_comp_d;
            face_analyser.pdm.princ_comp.convertTo(princ_comp_d, CV_64F);
            
            // locs = princ_comp (204x34) * local_params (34x1) = (204x1)
            Mat_<double> locs = princ_comp_d * local_params_row.t();
            
            // geom_descriptor_frame = [locs.t() (1x204) | local_params (1x34)] = 1x238
            Mat_<double> geom_desc;
            cv::hconcat(locs.t(), local_params_row, geom_desc);
            
            face_analyser.geom_descriptor_frame = geom_desc.clone();

            // 念のためGeomが横長であることを確認（縦長なら転置）
            if (face_analyser.geom_descriptor_frame.rows > face_analyser.geom_descriptor_frame.cols) {
                face_analyser.geom_descriptor_frame = face_analyser.geom_descriptor_frame.t();
            }

            // D. Timestamp
            face_analyser.current_time_seconds = (double)i * 0.033; 
            
            // ★重要: Dynamic model用のRunning Median更新
            // OpenFaceのAddNextFrameと同様に中央値を更新
            // これがないとdynamic modelが正しく機能しない
            face_analyser.frames_tracking++;
            
            // HOG medianを更新（2フレームに1回、高速化のため）
            if (face_analyser.frames_tracking % 2 == 1) {
                face_analyser.UpdateRunningMedian(
                    face_analyser.hog_desc_hist[0],  // view 0を使用
                    face_analyser.hog_hist_sum[0],
                    face_analyser.hog_desc_median,
                    face_analyser.hog_desc_frame,
                    true,  // update
                    face_analyser.num_bins_hog,
                    face_analyser.min_val_hog,
                    face_analyser.max_val_hog
                );
                face_analyser.hog_desc_median.setTo(0, face_analyser.hog_desc_median < 0);
                
                // Geom medianを更新
                face_analyser.UpdateRunningMedian(
                    face_analyser.geom_desc_hist,
                    face_analyser.geom_hist_sum,
                    face_analyser.geom_descriptor_median,
                    face_analyser.geom_descriptor_frame,
                    true,  // update
                    face_analyser.num_bins_geom,
                    face_analyser.min_val_geom,
                    face_analyser.max_val_geom
                );
            }

            // E. Predict
            if (i == 0) {
                 double min_h, max_h;
                 cv::minMaxLoc(face_analyser.hog_desc_frame, &min_h, &max_h);
                 Scalar mean_h = cv::mean(face_analyser.hog_desc_frame);
                 cout << "Debug: Frame 0 HOG Stats - Min: " << min_h << ", Max: " << max_h << ", Mean: " << mean_h[0] << endl;
                 cout << "Debug: HOG dims: " << face_analyser.hog_desc_frame.cols << " x " << face_analyser.hog_desc_frame.rows << endl;
                 cout << "Debug: Geom dims: " << face_analyser.geom_descriptor_frame.cols << " x " << face_analyser.geom_descriptor_frame.rows << endl;

                 // --- Geometry Descriptor Inspection ---
                 double min_g, max_g;
                 cv::minMaxLoc(face_analyser.geom_descriptor_frame, &min_g, &max_g);
                 Scalar mean_g = cv::mean(face_analyser.geom_descriptor_frame);
                 cout << "Debug: Frame 0 Geom Stats - Min: " << min_g << ", Max: " << max_g << ", Mean: " << mean_g[0] << endl;
                 
                 // Print first few values to see if they look like pixels or normalized params
                 cout << "Debug: Frame 0 Geom First 10 vals: ";
                 for(int k=0; k<10 && k<face_analyser.geom_descriptor_frame.cols; ++k) {
                     cout << face_analyser.geom_descriptor_frame.at<double>(0, k) << " ";
                 }
                 cout << endl;
                 // --------------------------------------
            }

            // PredictCurrentAUs returns the predictions directly, it does NOT populate AU_predictions_reg member
            // PredictCurrentAUsClass returns classifications, it does NOT populate AU_predictions_class member
            auto preds_r = face_analyser.PredictCurrentAUs(0);
            auto preds_c = face_analyser.PredictCurrentAUsClass(0);
            
            // if (i == 0) cout << "Debug: preds_r size=" << preds_r.size() << ", preds_c size=" << preds_c.size() << endl;

            // F. Write Result
            out_file << i << "," << face_analyser.current_time_seconds;
            
            for (const auto& p : preds_c) out_file << "," << p.second;
            for (auto& p : preds_r) {
                // Post-processing: Clamp to [0, 5]
                double val = p.second;
                if (val < 0.0) val = 0.0;
                if (val > 5.0) val = 5.0;
                out_file << "," << val;
            }
            
            out_file << endl;
        }

        // cout << "Debug: Loop finished." << endl;

    } catch (std::exception& e) {
        cerr << "Error: Exception in processing loop/initialization: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Error: Unknown exception occurred." << endl;
        return 1;
    }

    cout << "Done. Saved to " << output_file << endl;
    return 0;
}
