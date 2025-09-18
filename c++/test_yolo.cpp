#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>

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

int main() {
    std::cout << "=== YOLO Service C++ Test ===" << std::endl;
    
    // Initialize YOLO service
    YoloService yolo;
    
    // Initialize model (adjust path as needed)
    std::string engine_path = "./yolo11n.engine";
    if (!yolo.init(engine_path, 640, 0.25, 0)) {
        std::cerr << "Failed to initialize YOLO model" << std::endl;
        return -1;
    }
    
    std::cout << "YOLO model initialized successfully!" << std::endl;
    
    // Test with webcam or image file
    cv::VideoCapture cap(0);  // Use webcam, or change to image file path
    
    if (!cap.isOpened()) {
        std::cout << "Trying image file instead..." << std::endl;
        cap.open("test_image.jpg");  // Change to your test image
        if (!cap.isOpened()) {
            std::cerr << "Could not open camera or image file" << std::endl;
            return -1;
        }
    }
    
    std::cout << "Press 'q' to quit, 's' to save frame" << std::endl;
    
    cv::Mat frame;
    int frame_count = 0;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        // Run detection
        auto start = std::chrono::high_resolution_clock::now();
        std::string result = yolo.detect_cv_image(frame);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Parse and display result
        std::cout << "Frame " << frame_count++ 
                  << " - Detection time: " << duration.count() << "ms" 
                  << " - Result: " << result << std::endl;
        
        // Display frame
        cv::imshow("YOLO Test", frame);
        
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q') break;
        if (key == 's') {
            cv::imwrite("saved_frame.jpg", frame);
            std::cout << "Frame saved as saved_frame.jpg" << std::endl;
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    std::cout << "Test completed!" << std::endl;
    return 0;
}
