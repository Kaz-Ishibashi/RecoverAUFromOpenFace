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
    // Read header
    if (!getline(file, line)) {
        cerr << "Error: Landmark file appears empty." << endl;
        return false;
    }
    
    // Parse Header to find "x_0" and "y_0"
    int x_start = -1;
    int y_start = -1;
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
            col_idx++;
        }
    }

    if (x_start == -1 || y_start == -1) {
        cerr << "Error: Could not find 'x_0' or 'y_0' columns in CSV header." << endl;
        cerr << "Header was: " << line.substr(0, 100) << "..." << endl;
        return false;
    }

    cout << "Debug: Found landmarks at x_start=" << x_start << ", y_start=" << y_start << endl;

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

    // 2. Load Landmarks
    // cout << "Debug: Loading Landmarks file: " << landmark_file << endl;
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
            
            bool hog_success = hog_frames[i].good_frame;
            bool land_success = (cv::countNonZero(shape_2d) > 0); // Check if we have valid landmarks
            bool success = hog_success && land_success;

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
            
            // CP6: Raw Predictions
            vector<double> raw_reg_vals;
            for(size_t k=0; k<preds_r.size(); ++k) {
                DUMP_VAL(dump_frame_id, "CP6", k, preds_r[k].second);
                raw_reg_vals.push_back(preds_r[k].second);
            }
            vector<double> raw_class_vals;
            for(auto& p : preds_c) raw_class_vals.push_back(p.second);

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
        }

        // --- PHASE 2: OFFLINE CALIBRATION ---
        
        // A. Dynamic Shift (Baseline Subtraction) for REGRESSION
        vector<string> dyn_au_names = face_analyser.AU_SVR_dynamic_appearance_lin_regressors.GetAUNames();
        vector<double> cutoffs = face_analyser.AU_SVR_dynamic_appearance_lin_regressors.GetCutoffs();

        int num_reg_aus = face_analyser.GetAURegNames().size();
        
        // For each AU
        for (int au_idx = 0; au_idx < num_reg_aus; ++au_idx) {
            string au_name = face_analyser.GetAURegNames()[au_idx];
            
            // Check if dynamic
            bool is_dyn = false;
            int dyn_idx = -1;
            for(size_t d=0; d<dyn_au_names.size(); ++d) {
                if (dyn_au_names[d] == au_name) {
                    is_dyn = true;
                    dyn_idx = d;
                    break;
                }
            }

            double offset = 0.0;
            if (is_dyn && dyn_idx < cutoffs.size()) {
                // Collect valid predictions
                vector<double> valid_preds;
                for(const auto& h : history) {
                    if(h.success && au_idx < h.raw_reg.size()) valid_preds.push_back(h.raw_reg[au_idx]);
                }
                std::sort(valid_preds.begin(), valid_preds.end());
                
                double cutoff_ratio = cutoffs[dyn_idx];
                if (cutoff_ratio != -1 && !valid_preds.empty()) {
                     int idx = (int)(valid_preds.size() * cutoff_ratio);
                     if (idx < valid_preds.size()) offset = valid_preds[idx];
                }
            }
            
            // CP7: Offset
            DUMP_VAL(-1, "CP7", au_idx, offset);

            // Apply Offset & Clipping
            for(size_t i=0; i<history.size(); ++i) {
                if (history[i].success) {
                     double val = history[i].raw_reg[au_idx];
                     if (is_dyn) val -= offset;
                     
                     if (val < 0) val = 0;
                     if (val > 5) val = 5;
                     
                     history[i].final_reg[au_idx] = val;
                } else {
                     history[i].final_reg[au_idx] = 0;
                }
            }
            
            // Smoothing (Window=3)
            int window = 3;
            vector<double> smoothed(history.size());
            for(size_t i=(window-1)/2; i < history.size() - (window-1)/2; ++i) {
                double sum = 0;
                for(int w=-(window-1)/2; w <= (window-1)/2; ++w) {
                     sum += history[i+w].final_reg[au_idx];
                }
                history[i].final_reg[au_idx] = sum / window;
            }
            
            // CP8: Final Value
             for(size_t frame_i=0; frame_i<history.size(); ++frame_i) {
                  if (history[frame_i].success) // Only dump valid frames or all? OpenFace dumps all processed I think.
                     DUMP_VAL(frame_i + 1, "CP8", au_idx, history[frame_i].final_reg[au_idx]); // +1 for FrameID offset
             }
        }
        
        // Write Final Output to CSV
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
