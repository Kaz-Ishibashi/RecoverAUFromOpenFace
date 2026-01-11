#ifndef DUMP_LOGGER_H
#define DUMP_LOGGER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <mutex>
#include <opencv2/core/core.hpp>

// Simple singleton logger for debug dumps
class DumpLogger {
public:
    static DumpLogger& GetInstance() {
        static DumpLogger instance;
        return instance;
    }

    void Open(const std::string& filename) {
        if (outfile.is_open()) outfile.close();
        outfile.open(filename, std::ios::out | std::ios::trunc);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open dump file: " << filename << std::endl;
        } else {
            // outfile << "FrameID,CheckpointID,DimIndex,Value\n"; // No header for easier processing if preferred, but plan said header-less or use python names
            // Implementation plan implies strictly data? Python script uses "header=None".
            // So NO Header.
        }
    }

    void DumpVal(int frame_id, const std::string& cp_id, int dim_idx, double value) {
        if (!outfile.is_open()) return;
        // Format: FrameID, CheckpointID, DimensionIndex, Value (Precision 15)
        outfile << frame_id << "," << cp_id << "," << dim_idx << "," 
                << std::setprecision(15) << value << "\n";
    }

    // Helper for dumping cv::Mat (1D or 2D handled as flat sequence)
    void DumpMat(int frame_id, const std::string& cp_id, const cv::Mat& mat) {
        if (mat.empty()) return;
        
        cv::Mat flat = mat.reshape(1, 1); // Treat as 1D row
        if (flat.depth() == CV_64F) {
            for (int i = 0; i < flat.cols; ++i) {
                DumpVal(frame_id, cp_id, i, flat.at<double>(0, i));
            }
        } else if (flat.depth() == CV_32F) {
             for (int i = 0; i < flat.cols; ++i) {
                DumpVal(frame_id, cp_id, i, (double)flat.at<float>(0, i));
            }
        } else {
            // Handle other types if necessary (e.g. uchar for image sum)
             for (int i = 0; i < flat.cols; ++i) {
                 // Generic fallback
                 DumpVal(frame_id, cp_id, i, (double)flat.at<double>(0,i)); // Might crash if type wrong, assumes double/float mostly used in features
            }
        }
    }
    
    // Helper for vector
    void DumpVec(int frame_id, const std::string& cp_id, const std::vector<double>& vec) {
        for (size_t i = 0; i < vec.size(); ++i) {
            DumpVal(frame_id, cp_id, (int)i, vec[i]);
        }
    }

    ~DumpLogger() {
        if (outfile.is_open()) outfile.close();
    }

private:
    DumpLogger() {}
    std::ofstream outfile;
    DumpLogger(const DumpLogger&) = delete;
    void operator=(const DumpLogger&) = delete;
};

// Convenience macro
#define DUMP_VAL(frame, cp, dim, val) DumpLogger::GetInstance().DumpVal(frame, cp, dim, val)
#define DUMP_MAT(frame, cp, mat) DumpLogger::GetInstance().DumpMat(frame, cp, mat)
#define DUMP_VEC(frame, cp, vec) DumpLogger::GetInstance().DumpVec(frame, cp, vec)
#define INIT_DUMP(filename) DumpLogger::GetInstance().Open(filename)

#endif // DUMP_LOGGER_H
