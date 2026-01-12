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
#include "DumpLogger.h"

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

    // Data in file is already in the correct order (y, x, ch) as written by RecorderHOG.cpp
    // RecorderHOG writes: for y (0..cols) -> for x (0..rows) -> for o (0..31)
    // This matches the FaceAnalyser memory layout.
    // So we just need to cast from float (file) to double (memory).

    for (size_t k = 0; k < total_elements; ++k) {
        frame.hog_data(0, k) = (double)raw_data[k];
    }
    
    return true;
}


// -------------------------------------------------------------------------------------------------------------------
// Helper Function: Load Landmarks and Success flags from CSV
// ランドマークとsuccessフラグをCSVからロード
// -------------------------------------------------------------------------------------------------------------------
bool LoadLandmarks(const string& csv_path, vector<Mat_<float>>& all_landmarks, vector<bool>& success_flags) {
    cout << "Debug: Opening landmark file: " << csv_path << endl;
    ifstream file(csv_path);
    if (!file.is_open()) {
        cerr << "Error: Could not open landmark file: " << csv_path << endl;
        return false;
    }

    string line;
    // Read header
    if (!getline(file, line)) {
        cerr << "Error: Landmark file appears empty." << endl;
        return false;
    }
    
    // Parse Header to find "x_0", "y_0", and "success"
    int x_start = -1;
    int y_start = -1;
    int success_col = -1;
    {
        stringstream ss(line);
        string val_str;
        int col_idx = 0;
        while (getline(ss, val_str, ',')) {
            // Trim whitespace
            val_str.erase(0, val_str.find_first_not_of(" \t\r\n"));
            val_str.erase(val_str.find_last_not_of(" \t\r\n") + 1);
            
            if (val_str == "x_0") x_start = col_idx;
            if (val_str == "y_0") y_start = col_idx;
            if (val_str == "success") success_col = col_idx;
            col_idx++;
        }
    }

    if (x_start == -1 || y_start == -1) {
        cerr << "Error: Could not find 'x_0' or 'y_0' columns in CSV header." << endl;
        cerr << "Header was: " << line.substr(0, 100) << "..." << endl;
        return false;
    }
    
    if (success_col == -1) {
        cerr << "Warning: Could not find 'success' column. Assuming all frames are successful." << endl;
    }

    cout << "Debug: Found landmarks at x_start=" << x_start << ", y_start=" << y_start 
         << ", success_col=" << success_col << endl;

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
                     values.push_back(0.0f); // Default to 0 on error
                }
            } else {
                values.push_back(0.0f); // Handle empty fields
            }
        }

        // Check bounds
        // We need at least up to max(x_start, y_start) + 67
        int required_cols = std::max(x_start, y_start) + 68;
        if (values.size() < required_cols) {
            continue; 
        }

        Mat_<float> landmarks(2, 68);
        
        for(int i=0; i<68; ++i) {
            landmarks(0, i) = values[x_start + i]; // x_0 ... x_67
            landmarks(1, i) = values[y_start + i]; // y_0 ... y_67
        }
        
        all_landmarks.push_back(landmarks);
        
        // Read success flag (1 = success, 0 = failure)
        // successフラグを読み取る (1=成功, 0=失敗)
        bool success = true;  // Default to true if column not found
        if (success_col != -1 && success_col < values.size()) {
            success = (values[success_col] > 0.5);  // Treat > 0.5 as success
        }
        success_flags.push_back(success);
        
        row_count++;
    }
    
    int success_count = std::count(success_flags.begin(), success_flags.end(), true);
    cout << "Debug: Loaded " << row_count << " landmark frames (" << success_count << " successful)." << endl;
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

    INIT_DUMP("dump_recover.csv");

    // 1. Load HOG Data (Custom Format)
    // cout << "Debug: Loading HOG file: " << hog_file << endl;
    vector<RawHOGFrame> hog_frames;
    
    try {
        ifstream fin(hog_file, ios::binary);
        if (!fin) {
             cerr << "Error: HOG file not found or cannot be opened." << endl;
             return 1;
        }

        // cout << "Debug: Reading custom HOG frames..." << endl;
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

    // 2. Load Landmarks and Success Flags
    // ランドマークとsuccessフラグをロード
    vector<Mat_<float>> landmarks_data;
    vector<bool> csv_success_flags;  // Success flags from CSV file
    if (!LoadLandmarks(landmark_file, landmarks_data, csv_success_flags)) {
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
    // cout << "Debug: Initializing FaceAnalyser..." << endl;
    
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
    // fa_args.push_back("-au_static"); // Commented out to match default Dynamic behavior 
    
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

        // Structure to hold history for Offline Calibration
        struct FrameResult {
            double timestamp;
            bool success;
            vector<double> raw_reg;
            vector<double> raw_class;
            vector<double> final_reg;
            vector<double> final_class;
        };
        vector<FrameResult> history;
        
        // Storage for initial frame descriptors (for postprocessing like OpenFace)
        // 初期フレーム記述子の保存（OpenFaceと同様のポストプロセス用）
        vector<cv::Mat_<double>> hog_desc_frames_init;
        vector<cv::Mat_<double>> geom_descriptor_frames_init;
        int frames_tracking_succ = 0;
        int max_init_frames = 3000;  // Same as OpenFace

        // 4. Processing Loop
        // ========================================================================
        // NOTE: FrameID Offset for Dump Comparison
        // OpenFace's FaceAnalyser::AddNextFrame() has DIFFERENT offset for CP1 vs others:
        //   - CP1 is dumped BEFORE frames_tracking++ (Line 317 in FaceAnalyser.cpp)
        //   - CP2, CP3, CP4, etc. are dumped AFTER frames_tracking++ (Line 319+)
        // 
        // So for RecoverAU:
        //   - CP1: use i (GT FrameID 0-215)
        //   - All other CPs: use i+1 (GT FrameID 1-216)
        // ========================================================================
        cout << "Debug: Starting processing loop..." << endl;
        
        for (size_t i = 0; i < num_frames; ++i) {
            // FrameID for CP1: use i (matches GT's pre-increment behavior)
            // FrameID for all other CPs: use i+1 (matches GT's post-increment behavior)
            // CP1用FrameID: i を使用 (GTのインクリメント前の動作に一致)
            // 他のCP用FrameID: i+1 を使用 (GTのインクリメント後の動作に一致)
            int dump_frame_id_cp1 = static_cast<int>(i);      // For CP1 only
            int dump_frame_id = static_cast<int>(i) + 1;       // For CP2, CP3, CP4, etc.

            // A. HOG
            // Set dimensions
            face_analyser.num_hog_rows = hog_frames[i].num_rows;
            face_analyser.num_hog_cols = hog_frames[i].num_cols;
            
            // HOG Data: Convert to standard Row Vector (1 x N)
            // FaceAnalyser expects Row Vectors for feature concatenation (hconcat)
            face_analyser.hog_desc_frame = hog_frames[i].hog_data.clone(); 
            
            // B. Calculate Geometry
            Mat_<float> shape_2d = landmarks_data[i];
            
            // CP1: Raw Landmarks (uses pre-increment FrameID)
            DUMP_MAT(dump_frame_id_cp1, "CP1", shape_2d);

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
            
            // -----------------------------------------------------------------------
            // PDM Parameter Calculation (The Core "Normalization" Step)
            // -----------------------------------------------------------------------
            
            // Format landmarks for PDM::CalcParams: Column vector (2*n x 1), [x0...xn, y0...yn]
            Mat_<float> shape_2d_formatted(face_analyser.pdm.NumberOfPoints() * 2, 1);
            for(int k=0; k<face_analyser.pdm.NumberOfPoints(); ++k) {
                shape_2d_formatted(k, 0) = shape_2d(0, k); // x
                shape_2d_formatted(k + face_analyser.pdm.NumberOfPoints(), 0) = shape_2d(1, k); // y
            }
            
            Vec6f params_global;
            Mat_<float> params_local;
            
            face_analyser.pdm.CalcParams(params_global, params_local, shape_2d_formatted);

            // CP9: Params Global (Rigid)
            for(int k=0; k<6; ++k) DUMP_VAL(dump_frame_id, "CP9", k, params_global[k]);
            
            // CP10: Params Local (Non-Rigid)
            DUMP_MAT(dump_frame_id, "CP10", params_local);
            
            if (i == 0) {
                double min_p, max_p;
                cv::minMaxLoc(params_local, &min_p, &max_p);
                cout << "Debug: Frame 0 params_local - Min: " << min_p << ", Max: " << max_p << endl;
                cout << "Debug: Frame 0 params_global - Scale: " << params_global[0] << ", Tx: " << params_global[4] << ", Ty: " << params_global[5] << endl;
                cout << "Debug: Landmarks (formatted) First 4: " << shape_2d_formatted(0,0) << ", " << shape_2d_formatted(1,0) << ", " << shape_2d_formatted(68,0) << ", " << shape_2d_formatted(69,0) << endl; 
            }

            // geom_descriptor construction: [locs | local_params]
            // Note: PDM in OpenFace is 3D (X,Y,Z), so princ_comp is (3*n x num_params) -> (204 x 34)
            
            // params_local (Col Vector) -> Row Vector
            Mat_<double> local_params_row;
            params_local.convertTo(local_params_row, CV_64F);
            local_params_row = local_params_row.t(); // 1 x 34
            
            // princ_comp conversion
            Mat_<double> princ_comp_d;
            face_analyser.pdm.princ_comp.convertTo(princ_comp_d, CV_64F);
            
            // Calculate shape deviations (non-rigid)
            Mat_<double> locs = princ_comp_d * local_params_row.t(); // (204x34) * (34x1) = (204x1)
            
            // Concatenate
            Mat_<double> geom_desc;
            cv::hconcat(locs.t(), local_params_row, geom_desc);
            
            face_analyser.geom_descriptor_frame = geom_desc.clone();
            
            // Ensure Row vector (1 x 238)
            if (face_analyser.geom_descriptor_frame.rows > face_analyser.geom_descriptor_frame.cols) {
                face_analyser.geom_descriptor_frame = face_analyser.geom_descriptor_frame.t();
            }

            // D. Timestamp
            face_analyser.current_time_seconds = (double)i * 0.033; 
            
            // Running Median Updates (Critical for Dynamic Models)
            // OpenFace logic: Update median every frame (or decimated) IF SUCCESSFUL
            
            // Use success flag from CSV (matches OpenFace's face tracking success)
            // CSVからのsuccessフラグを使用（OpenFaceの顔追跡成功に一致）
            bool success = (i < csv_success_flags.size()) ? csv_success_flags[i] : true;

            face_analyser.frames_tracking++;
            
            if (success && (face_analyser.frames_tracking % 2 == 1)) {
                // Update HOG median
                face_analyser.UpdateRunningMedian(
                    face_analyser.hog_desc_hist[0],
                    face_analyser.hog_hist_sum[0],
                    face_analyser.hog_desc_median,
                    face_analyser.hog_desc_frame,
                    true,
                    face_analyser.num_bins_hog,
                    face_analyser.min_val_hog,
                    face_analyser.max_val_hog
                );
                face_analyser.hog_desc_median.setTo(0, face_analyser.hog_desc_median < 0);
                
                // Update Geom median
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
                DUMP_MAT(dump_frame_id, "CP5_HOG", face_analyser.hog_desc_median);
                DUMP_MAT(dump_frame_id, "CP5_Geom", face_analyser.geom_descriptor_median);
            }
            
            // Dump HOG/Geom (CP3/CP4)
            if (dump_frame_id <= 5) DUMP_MAT(dump_frame_id, "CP3", face_analyser.hog_desc_frame); // HOG CP3
            DUMP_MAT(dump_frame_id, "CP4", face_analyser.geom_descriptor_frame); // Geom CP4

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

            // PredictCurrentAUs returns the predictions directly
            auto preds_r = face_analyser.PredictCurrentAUs(0);
            auto preds_c = face_analyser.PredictCurrentAUsClass(0);
            
            // CP6: Raw Predictions (Regression)
            vector<double> raw_reg_vals;
            for(size_t k=0; k<preds_r.size(); ++k) {
                DUMP_VAL(dump_frame_id, "CP6", k, preds_r[k].second);
                raw_reg_vals.push_back(preds_r[k].second);
            }
            
            // CP11: Raw Classification
            vector<double> raw_class_vals;
            for(size_t k=0; k<preds_c.size(); ++k) {
                DUMP_VAL(dump_frame_id, "CP11", k, preds_c[k].second);
                raw_class_vals.push_back(preds_c[k].second);
            }



            // Store in history (don't write yet)
            FrameResult res;
            res.timestamp = face_analyser.current_time_seconds;
            res.success = success;
            res.raw_reg = raw_reg_vals;
            res.raw_class = raw_class_vals;
            // Initialize final with raw for now
            res.final_reg = raw_reg_vals; 
            res.final_class = raw_class_vals;
            history.push_back(res);
            
            // Store initial frame descriptors for postprocessing (like OpenFace)
            // 初期フレーム記述子を保存（OpenFaceと同様のポストプロセス用）
            if (success && frames_tracking_succ < max_init_frames) {
                hog_desc_frames_init.push_back(face_analyser.hog_desc_frame.clone());
                geom_descriptor_frames_init.push_back(face_analyser.geom_descriptor_frame.clone());
                frames_tracking_succ++;
            }
        }
        
        // --- PHASE 1.5: POSTPROCESSING (Re-compute initial frame predictions with final medians) ---
        // OpenFace does this in PostprocessPredictions() before offset calculation
        // This is critical for matching CP7 offsets!
        // OpenFaceはPostprocessPredictions()でオフセット計算前にこれを行う
        cout << "Debug: Postprocessing " << frames_tracking_succ << " initial frames with final medians..." << endl;
        {
            int success_ind = 0;
            int all_ind = 0;
            
            while(all_ind < (int)history.size() && success_ind < frames_tracking_succ) {
                if(history[all_ind].success) {
                    // Restore descriptors from stored initial frames
                    face_analyser.hog_desc_frame = hog_desc_frames_init[success_ind].clone();
                    face_analyser.geom_descriptor_frame = geom_descriptor_frames_init[success_ind].clone();
                    
                    // Fix orientation if needed
                    if (face_analyser.hog_desc_frame.rows > face_analyser.hog_desc_frame.cols) {
                        face_analyser.hog_desc_frame = face_analyser.hog_desc_frame.t();
                    }
                    if (face_analyser.geom_descriptor_frame.rows > face_analyser.geom_descriptor_frame.cols) {
                        face_analyser.geom_descriptor_frame = face_analyser.geom_descriptor_frame.t();
                    }
                    
                    // Re-compute predictions with final (stabilized) medians
                    auto preds_r_new = face_analyser.PredictCurrentAUs(0);
                    auto preds_c_new = face_analyser.PredictCurrentAUsClass(0);

                    // Update history with new predictions
                    // Regression
                    for(size_t k = 0; k < preds_r_new.size() && k < history[all_ind].raw_reg.size(); ++k) {
                        history[all_ind].raw_reg[k] = preds_r_new[k].second;
                        history[all_ind].final_reg[k] = preds_r_new[k].second;
                    }
                    // Classification
                    for(size_t k = 0; k < preds_c_new.size() && k < history[all_ind].raw_class.size(); ++k) {
                        history[all_ind].raw_class[k] = preds_c_new[k].second;
                        history[all_ind].final_class[k] = preds_c_new[k].second;
                    }
                    
                    success_ind++;
                }
                all_ind++;
            }
        }
        cout << "Debug: Postprocessing complete." << endl;

        // --- PHASE 2: OFFLINE CALIBRATION ---
        // OpenFace's ExtractAllPredictionsOfflineReg iterates over AU_predictions_reg_all_hist
        // which is a std::map<string, ...> - meaning it's sorted ALPHABETICALLY by AU name!
        // We must iterate in the same order to match CP7/CP8 dump indices.
        // OpenFaceのExtractAllPredictionsOfflineRegは std::map を反復するため
        // AU名でアルファベット順にソートされている。同じ順序で反復する必要がある。
        
        // A. Get all AU names and sort them alphabetically
        vector<string> all_au_names = face_analyser.GetAURegNames();
        vector<string> sorted_au_names = all_au_names;
        std::sort(sorted_au_names.begin(), sorted_au_names.end());
        
        // Create mapping: sorted index -> original index (for accessing raw_reg)
        // ソート済みインデックス -> 元のインデックスへのマッピング作成
        vector<int> sorted_to_orig(sorted_au_names.size());
        for(size_t sorted_idx = 0; sorted_idx < sorted_au_names.size(); ++sorted_idx) {
            for(size_t orig_idx = 0; orig_idx < all_au_names.size(); ++orig_idx) {
                if(sorted_au_names[sorted_idx] == all_au_names[orig_idx]) {
                    sorted_to_orig[sorted_idx] = orig_idx;
                    break;
                }
            }
        }
        
        // B. Get dynamic AU names and cutoffs for offset calculation
        vector<string> dyn_au_names = face_analyser.AU_SVR_dynamic_appearance_lin_regressors.GetAUNames();
        vector<double> cutoffs = face_analyser.AU_SVR_dynamic_appearance_lin_regressors.GetCutoffs();

        // C. Calculate offsets in sorted (alphabetical) order - matching OpenFace's map iteration
        // OpenFaceのマップ反復に合わせて、ソートされた順序（アルファベット順）でオフセットを計算
        vector<double> offsets;
        int cp7_dump_idx = 0;
        
        for (size_t sorted_idx = 0; sorted_idx < sorted_au_names.size(); ++sorted_idx) {
            string au_name = sorted_au_names[sorted_idx];
            int orig_idx = sorted_to_orig[sorted_idx];  // For accessing raw_reg
            
            // DEBUG: Print AU name and indices for first few
            if(sorted_idx < 5) {
                cout << "Debug CP7: sorted_idx=" << sorted_idx << " au_name=" << au_name 
                     << " orig_idx=" << orig_idx << endl;
            }
            
            // Collect valid predictions for this AU
            vector<double> au_good;
            for(const auto& h : history) {
                if(h.success && orig_idx < h.raw_reg.size()) {
                    au_good.push_back(h.raw_reg[orig_idx]);
                }
            }
            
            // OpenFace logic (line 614-644):
            // if(au_good.empty() || !dynamic) { offsets.push_back(0); } // No dump
            // else { ... DUMP_VAL after offset calculation ... }
            
            // Check if this AU is dynamic
            bool is_dynamic_param = true;  // In OpenFace, 'dynamic' param is true when called from FaceLandmarkVid
            
            if(au_good.empty() || !is_dynamic_param) {
                offsets.push_back(0.0);
                // No CP7 dump in this path
            } else {
                // This AU enters the else branch - will always get a CP7 dump
                std::sort(au_good.begin(), au_good.end());
                
                // Find if this AU is in dynamic regressors
                int au_id = -1;
                for(size_t d = 0; d < dyn_au_names.size(); ++d) {
                    if(au_name == dyn_au_names[d]) {
                        au_id = d;
                        break;
                    }
                }
                
                double offset = 0.0;
                if(au_id != -1 && au_id < cutoffs.size() && cutoffs[au_id] != -1) {
                    double cutoff_ratio = cutoffs[au_id];
                    int idx = (int)(au_good.size() * cutoff_ratio);
                    if(idx < au_good.size()) offset = au_good[idx];
                    
                    // DEBUG: Print cutoff lookup
                    if(sorted_idx < 5) {
                        cout << "  Found in dyn_au_names at " << au_id << ", cutoff=" << cutoff_ratio 
                             << ", au_good.size=" << au_good.size() << ", idx=" << idx << ", offset=" << offset << endl;
                    }
                } else {
                    // DEBUG: Print why offset is 0
                    if(sorted_idx < 5) {
                        cout << "  au_id=" << au_id << " (not found or cutoff=-1), offset=0" << endl;
                    }
                }
                // else: offset stays 0.0
                
                offsets.push_back(offset);
                
                // CP7: Dump for ALL AUs that enter the else branch
                // OpenFace dumps at line 643 which is AFTER the inner if/else
                // CP7はelse分岐に入るすべてのAUでダンプ
                DUMP_VAL(-1, "CP7", cp7_dump_idx, offset);
                cp7_dump_idx++;
            }
        }
        
        // D. Apply offsets and dump CP8 (in sorted order, before smoothing)
        // Note: offsets[sorted_idx] corresponds to AU at sorted_au_names[sorted_idx]
        for (size_t sorted_idx = 0; sorted_idx < sorted_au_names.size(); ++sorted_idx) {
            int orig_idx = sorted_to_orig[sorted_idx];
            string au_name = sorted_au_names[sorted_idx];
            
            // Check if this AU is dynamic (for offset subtraction)
            bool is_dyn = false;
            for(const auto& dyn_name : dyn_au_names) {
                if(dyn_name == au_name) {
                    is_dyn = true;
                    break;
                }
            }
            
            double offset = offsets[sorted_idx];
            
            for(size_t frame_i = 0; frame_i < history.size(); ++frame_i) {
                if(history[frame_i].success) {
                    double val = history[frame_i].raw_reg[orig_idx];
                    val = (val - offset) * 1.0;  // scaling = 1
                    
                    if(val < 0.0) val = 0.0;
                    if(val > 5.0) val = 5.0;
                    
                    history[frame_i].final_reg[orig_idx] = val;
                    
                    // CP8: Dump with frame_i (0-indexed, like OpenFace's loop variable 'frame')
                    // OpenFace: DUMP_VAL(frame, "CP8", ...) where frame is 0-indexed loop counter
                    // CP8はframe_iを使用（OpenFaceの'frame'変数と同じ0-indexed）
                    DUMP_VAL(frame_i, "CP8", sorted_idx, val);  // No +1, use 0-indexed like OpenFace
                } else {
                    history[frame_i].final_reg[orig_idx] = 0;
                }
            }
        }
        
        // E. Smoothing (in sorted order)
        for (size_t sorted_idx = 0; sorted_idx < sorted_au_names.size(); ++sorted_idx) {
            int orig_idx = sorted_to_orig[sorted_idx];
            
            int window = 3;
            vector<double> au_vals_tmp(history.size());
            for(size_t i = 0; i < history.size(); ++i) {
                au_vals_tmp[i] = history[i].final_reg[orig_idx];
            }
            for(size_t i=(window-1)/2; i < history.size() - (window-1)/2; ++i) {
                double sum = 0;
                for(int w=-(window-1)/2; w <= (window-1)/2; ++w) {
                     sum += au_vals_tmp[i+w];
                }
                history[i].final_reg[orig_idx] = sum / window;
            }
        }
        
        // Write Final Output to CSV
        
        // --- PHASE 3: SMOOTHING FOR CLASSIFICATION (WINDOW=7) ---
        // Matches FaceAnalyser::ExtractAllPredictionsOfflineClass logic
        // 1. Moving Average (size 7)
        // 2. Threshold at 0.5 (0 or 1)
        
        // A. Get Class Names and Sort them to match OpenFace CP12 indexing
        if(!history.empty()) {
            vector<string> raw_class_names = face_analyser.GetAUClassNames();
            vector<string> sorted_class_names = raw_class_names; // Copy
            std::sort(sorted_class_names.begin(), sorted_class_names.end());
            
            // Map sorted index to original index
            map<int, int> sorted_to_orig_class;
            for(size_t i=0; i<sorted_class_names.size(); ++i) {
                for(size_t j=0; j<raw_class_names.size(); ++j) {
                    if(sorted_class_names[i] == raw_class_names[j]) {
                        sorted_to_orig_class[i] = j;
                        break;
                    }
                }
            }

            size_t num_class_aus = raw_class_names.size();
            
            // Iterate in SORTED order for CP12 consistency
            for(size_t sorted_idx = 0; sorted_idx < num_class_aus; ++sorted_idx) {
                int orig_idx = sorted_to_orig_class[sorted_idx];
                
                // Extract time series for this AU (using orig_idx)
                vector<double> series;
                for(const auto& h : history) {
                    if(orig_idx < h.raw_class.size()) series.push_back(h.raw_class[orig_idx]);
                    else series.push_back(0.0);
                }
                
                // Apply smoothing
                vector<double> smoothed = series;
                int window_size = 7;
                if((int)series.size() > (window_size - 1) / 2) {
                    for (size_t i = (window_size - 1)/2; i < series.size() - (window_size - 1) / 2; ++i) {
                        double sum = 0;
                        int div_by = 0;
                        for (int w = -(window_size - 1) / 2; w <= (window_size - 1) / 2; ++w) {
                            if(i+w < series.size()) {
                                sum += series[i+w];
                                div_by++;
                            }
                        }
                        sum = sum / div_by;
                        if (sum < 0.5) sum = 0;
                        else sum = 1;
                        
                        smoothed[i] = sum;
                    }
                }
                
                // Write back to history (using orig_idx) and Dump CP12 (using sorted_idx)
                for(size_t f=0; f<history.size(); ++f) {
                    // Update final_class
                    if(orig_idx < history[f].final_class.size()) {
                        history[f].final_class[orig_idx] = smoothed[f];
                    }
                    
                    // CP12: Final Classification
                    // Index must be sorted_idx to match OpenFace's map iteration!
                    if(history[f].success) {
                        DUMP_VAL((int)f, "CP12", (int)sorted_idx, smoothed[f]);
                    } else {
                        DUMP_VAL((int)f, "CP12", (int)sorted_idx, 0.0);
                    }
                }
            }
        }

        for(size_t i=0; i<history.size(); ++i) {
            out_file << i << "," << history[i].timestamp;
            for(double v : history[i].final_class) out_file << "," << v; // Class not corrected here for brevity
            for(double v : history[i].final_reg) out_file << "," << v;
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
