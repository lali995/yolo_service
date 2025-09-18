#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace py = pybind11;

class YoloService {
private:
    py::scoped_interpreter guard{};
    py::object mod, init_fn, detect_fn, shutdown_fn, err_fn;
    bool initialized = false;

public:
    YoloService() {
        try {
            mod = py::module_::import("yolo_service");
            init_fn = mod.attr("init_engine");
            detect_fn = mod.attr("detect_image");
            shutdown_fn = mod.attr("shutdown");
            err_fn = mod.attr("get_last_error");
        } catch (const std::exception& e) {
            std::cerr << "Failed to import yolo_service: " << e.what() << std::endl;
        }
    }

    bool init(const std::string& engine_path, int imgsz = 640, float conf = 0.25, int device = 0) {
        try {
            bool result = init_fn(engine_path, imgsz, conf, device, true).cast<bool>();
            initialized = result;
            if (!result) {
                std::string error = err_fn().cast<std::string>();
                std::cerr << "Init failed: " << error << std::endl;
            }
            return result;
        } catch (const std::exception& e) {
            std::cerr << "Exception during init: " << e.what() << std::endl;
            return false;
        }
    }

    std::string detect_cv_image(cv::Mat& image) {
        if (!initialized) {
            return R"({"ok": false, "error": "not_initialized"})";
        }

        try {
            // Convert OpenCV Mat to NumPy array - Simplified approach
            py::array_t<uint8_t> arr(
                {image.rows, image.cols, image.channels()},  // shape
                image.data                                   // data pointer
            );
            
            return detect_fn(arr).cast<std::string>();
        } catch (const std::exception& e) {
            std::cerr << "Exception during detection: " << e.what() << std::endl;
            return R"({"ok": false, "error": "detection_failed"})";
        }
    }

    void shutdown() {
        if (initialized) {
            shutdown_fn();
            initialized = false;
        }
    }

    ~YoloService() {
        shutdown();
    }
};

class HeadlessProcessor {
private:
    YoloService& yolo;
    std::string output_dir;
    int frame_count = 0;
    double total_processing_time = 0.0;
    int total_detections = 0;

public:
    HeadlessProcessor(YoloService& yolo_service, const std::string& output_directory = "output") 
        : yolo(yolo_service), output_dir(output_directory) {
        // Create output directory if it doesn't exist
        std::filesystem::create_directories(output_dir);
    }

    void process_image(const cv::Mat& frame, bool save_annotated = false) {
        cv::Mat frame_copy = frame.clone();
        
        // Run detection
        auto start = std::chrono::high_resolution_clock::now();
        std::string result = yolo.detect_cv_image(frame_copy);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double processing_time_ms = duration.count();
        total_processing_time += processing_time_ms;
        
        // Parse result to get detection count
        int detections = 0;
        if (result.find("\"count\":") != std::string::npos) {
            size_t start_pos = result.find("\"count\":") + 8;
            size_t end_pos = result.find(",", start_pos);
            if (end_pos == std::string::npos) end_pos = result.find("}", start_pos);
            if (end_pos != std::string::npos) {
                std::string count_str = result.substr(start_pos, end_pos - start_pos);
                detections = std::stoi(count_str);
            }
        }
        total_detections += detections;
        
        // Display result
        std::cout << "Frame " << frame_count++ 
                  << " - Detection time: " << processing_time_ms << "ms" 
                  << " - Detections: " << detections
                  << " - Result: " << result << std::endl;
        
        // Save annotated frame if requested
        if (save_annotated) {
            std::string filename = output_dir + "/frame_" + 
                                 std::to_string(frame_count - 1) + "_detections_" + 
                                 std::to_string(detections) + ".jpg";
            cv::imwrite(filename, frame_copy);
        }
    }

    void process_video(const std::string& video_path, bool save_annotated = false, int max_frames = -1) {
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Could not open video file: " << video_path << std::endl;
            return;
        }

        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        
        std::cout << "Processing video: " << video_path << std::endl;
        std::cout << "Total frames: " << total_frames << ", FPS: " << fps << std::endl;
        std::cout << "Max frames to process: " << (max_frames > 0 ? std::to_string(max_frames) : "all") << std::endl;
        std::cout << "Save annotated frames: " << (save_annotated ? "yes" : "no") << std::endl;
        std::cout << "Output directory: " << output_dir << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        cv::Mat frame;
        int processed_frames = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (cap.read(frame) && (max_frames == -1 || processed_frames < max_frames)) {
            process_image(frame, save_annotated);
            processed_frames++;
            
            // Progress update every 10 frames
            if (processed_frames % 10 == 0) {
                double progress = (double)processed_frames / total_frames * 100.0;
                std::cout << "Progress: " << std::fixed << std::setprecision(1) 
                         << progress << "% (" << processed_frames << "/" << total_frames << ")" << std::endl;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        cap.release();
        
        // Print summary
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Processing completed!" << std::endl;
        std::cout << "Processed frames: " << processed_frames << std::endl;
        std::cout << "Total processing time: " << total_duration.count() << "ms" << std::endl;
        std::cout << "Average processing time per frame: " 
                  << (total_processing_time / processed_frames) << "ms" << std::endl;
        std::cout << "Total detections: " << total_detections << std::endl;
        std::cout << "Average detections per frame: " 
                  << (double)total_detections / processed_frames << std::endl;
        if (save_annotated) {
            std::cout << "Annotated frames saved to: " << output_dir << std::endl;
        }
    }

    void process_single_image(const std::string& image_path, bool save_annotated = false) {
        cv::Mat frame = cv::imread(image_path);
        if (frame.empty()) {
            std::cerr << "Could not load image: " << image_path << std::endl;
            return;
        }

        std::cout << "Processing image: " << image_path << std::endl;
        std::cout << "Image size: " << frame.cols << "x" << frame.rows << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        process_image(frame, save_annotated);
        
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Image processing completed!" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "=== YOLO Service C++ Headless Test ===" << std::endl;
    
    // Parse command line arguments
    std::string engine_path = "./yolo11n.engine";
    std::string input_path = "test_image.jpg";
    std::string output_dir = "output";
    bool save_annotated = false;
    int max_frames = -1;
    bool is_video = false;

    if (argc > 1) {
        input_path = argv[1];
        // Check if it's a video file by extension
        std::string ext = input_path.substr(input_path.find_last_of(".") + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        is_video = (ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv");
    }
    if (argc > 2) {
        output_dir = argv[2];
    }
    if (argc > 3) {
        save_annotated = (std::string(argv[3]) == "true" || std::string(argv[3]) == "1");
    }
    if (argc > 4) {
        max_frames = std::stoi(argv[4]);
    }
    
    // Initialize YOLO service
    YoloService yolo;
    
    if (!yolo.init(engine_path, 640, 0.25, 0)) {
        std::cerr << "Failed to initialize YOLO model" << std::endl;
        return -1;
    }
    
    std::cout << "YOLO model initialized successfully!" << std::endl;
    
    // Create headless processor
    HeadlessProcessor processor(yolo, output_dir);
    
    // Process input
    if (is_video) {
        processor.process_video(input_path, save_annotated, max_frames);
    } else {
        processor.process_single_image(input_path, save_annotated);
    }
    
    std::cout << "Headless test completed!" << std::endl;
    return 0;
}
