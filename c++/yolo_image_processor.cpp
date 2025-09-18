#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace py = pybind11;

class YoloRgbaService {
private:
    py::scoped_interpreter guard{};
    py::object mod, init_fn, detect_rgba_fn, shutdown_fn, err_fn;
    bool initialized = false;

public:
    YoloRgbaService() {
        try {
            mod = py::module_::import("yolo_service");
            init_fn = mod.attr("init_engine");
            detect_rgba_fn = mod.attr("detect_rgba_image");
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

    std::string detect_rgba_data(const std::vector<uint8_t>& rgba_data, int width, int height) {
        if (!initialized) {
            return R"({"ok": false, "error": "not_initialized"})";
        }

        try {
            // Convert vector to numpy array
            py::array_t<uint8_t> arr(
                {height, width, 4},  // shape for RGBA
                rgba_data.data()     // data pointer
            );
            
            return detect_rgba_fn(arr, width, height).cast<std::string>();
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

    ~YoloRgbaService() {
        shutdown();
    }
};

struct Detection {
    int class_id;
    std::string class_name;
    float confidence;
    float x1, y1, x2, y2;
    float width, height;
};

class JsonParser {
public:
    static std::vector<Detection> parse_detections(const std::string& json_response) {
        std::vector<Detection> detections;
        
        try {
            // Simple JSON parsing for our specific format
            // This is a basic parser - in production you might want to use a proper JSON library
            
            if (json_response.find("\"ok\": false") != std::string::npos) {
                std::cerr << "Detection failed in JSON response" << std::endl;
                return detections;
            }
            
            // Find detections array
            size_t detections_start = json_response.find("\"detections\":");
            if (detections_start == std::string::npos) {
                return detections;
            }
            
            detections_start = json_response.find("[", detections_start);
            if (detections_start == std::string::npos) {
                return detections;
            }
            
            // Parse each detection object
            size_t pos = detections_start + 1;
            while (pos < json_response.length()) {
                size_t obj_start = json_response.find("{", pos);
                if (obj_start == std::string::npos) break;
                
                size_t obj_end = json_response.find("}", obj_start);
                if (obj_end == std::string::npos) break;
                
                std::string obj_str = json_response.substr(obj_start, obj_end - obj_start + 1);
                Detection det = parse_detection_object(obj_str);
                if (det.class_id >= 0) {  // Valid detection
                    detections.push_back(det);
                }
                
                pos = obj_end + 1;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Failed to parse detection JSON: " << e.what() << std::endl;
        }
        
        return detections;
    }

private:
    static Detection parse_detection_object(const std::string& obj_str) {
        Detection det = {-1, "", 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        
        // Parse class_id
        size_t pos = obj_str.find("\"class_id\":");
        if (pos != std::string::npos) {
            pos = obj_str.find(":", pos) + 1;
            while (pos < obj_str.length() && (obj_str[pos] == ' ' || obj_str[pos] == '\t')) pos++;
            size_t end = pos;
            while (end < obj_str.length() && obj_str[end] != ',' && obj_str[end] != '}') end++;
            det.class_id = std::stoi(obj_str.substr(pos, end - pos));
        }
        
        // Parse class_name
        pos = obj_str.find("\"class_name\":");
        if (pos != std::string::npos) {
            pos = obj_str.find("\"", pos) + 1;
            size_t end = obj_str.find("\"", pos);
            if (end != std::string::npos) {
                det.class_name = obj_str.substr(pos, end - pos);
            }
        }
        
        // Parse confidence
        pos = obj_str.find("\"confidence\":");
        if (pos != std::string::npos) {
            pos = obj_str.find(":", pos) + 1;
            while (pos < obj_str.length() && (obj_str[pos] == ' ' || obj_str[pos] == '\t')) pos++;
            size_t end = pos;
            while (end < obj_str.length() && obj_str[end] != ',' && obj_str[end] != '}') end++;
            det.confidence = std::stof(obj_str.substr(pos, end - pos));
        }
        
        // Parse bbox
        pos = obj_str.find("\"bbox\":");
        if (pos != std::string::npos) {
            pos = obj_str.find("{", pos);
            size_t bbox_end = obj_str.find("}", pos);
            if (bbox_end != std::string::npos) {
                std::string bbox_str = obj_str.substr(pos, bbox_end - pos + 1);
                
                // Parse x1, y1, x2, y2
                det.x1 = parse_float_value(bbox_str, "\"x1\":");
                det.y1 = parse_float_value(bbox_str, "\"y1\":");
                det.x2 = parse_float_value(bbox_str, "\"x2\":");
                det.y2 = parse_float_value(bbox_str, "\"y2\":");
                det.width = parse_float_value(bbox_str, "\"width\":");
                det.height = parse_float_value(bbox_str, "\"height\":");
            }
        }
        
        return det;
    }
    
    static float parse_float_value(const std::string& str, const std::string& key) {
        size_t pos = str.find(key);
        if (pos == std::string::npos) return 0.0f;
        
        pos = str.find(":", pos) + 1;
        while (pos < str.length() && (str[pos] == ' ' || str[pos] == '\t')) pos++;
        size_t end = pos;
        while (end < str.length() && str[end] != ',' && str[end] != '}') end++;
        
        return std::stof(str.substr(pos, end - pos));
    }
};

class ImageProcessor {
private:
    YoloRgbaService& yolo_service;
    std::string output_dir;

public:
    ImageProcessor(YoloRgbaService& service, const std::string& output_directory = "output") 
        : yolo_service(service), output_dir(output_directory) {
        std::filesystem::create_directories(output_dir);
    }

    bool process_image(const std::string& input_path, const std::string& output_path) {
        std::cout << "Processing image: " << input_path << std::endl;
        
        // Load image
        cv::Mat image = cv::imread(input_path, cv::IMREAD_UNCHANGED);
        if (image.empty()) {
            std::cerr << "Could not load image: " << input_path << std::endl;
            return false;
        }
        
        std::cout << "Image loaded: " << image.cols << "x" << image.rows << " channels: " << image.channels() << std::endl;
        
        // Convert to RGBA
        cv::Mat rgba_image;
        if (image.channels() == 3) {
            cv::cvtColor(image, rgba_image, cv::COLOR_BGR2RGBA);
        } else if (image.channels() == 4) {
            rgba_image = image;
        } else {
            std::cerr << "Unsupported image format. Expected 3 or 4 channels, got " << image.channels() << std::endl;
            return false;
        }
        
        std::cout << "Converted to RGBA: " << rgba_image.cols << "x" << rgba_image.rows << std::endl;
        
        // Convert RGBA to vector for YOLO service
        std::vector<uint8_t> rgba_data;
        rgba_data.assign(rgba_image.data, rgba_image.data + rgba_image.total() * rgba_image.elemSize());
        
        std::cout << "RGBA data size: " << rgba_data.size() << " bytes" << std::endl;
        
        // Run YOLO detection
        std::cout << "Running YOLO detection..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        std::string result_json = yolo_service.detect_rgba_data(rgba_data, rgba_image.cols, rgba_image.rows);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Detection completed in: " << duration.count() << "ms" << std::endl;
        
        // Parse results
        std::vector<Detection> detections = JsonParser::parse_detections(result_json);
        std::cout << "Found " << detections.size() << " detections" << std::endl;
        
        // Print detection details
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];
            std::cout << "  " << i + 1 << ". " << det.class_name 
                      << " (ID: " << det.class_id << ")"
                      << " - Confidence: " << std::fixed << std::setprecision(3) << det.confidence
                      << " - BBox: (" << det.x1 << ", " << det.y1 << ", " << det.x2 << ", " << det.y2 << ")"
                      << " - Size: " << det.width << "x" << det.height << std::endl;
        }
        
        // Render bounding boxes on original image
        cv::Mat result_image = image.clone();
        render_detections(result_image, detections);
        
        // Save result
        bool saved = cv::imwrite(output_path, result_image);
        if (saved) {
            std::cout << "Result saved to: " << output_path << std::endl;
        } else {
            std::cerr << "Failed to save result to: " << output_path << std::endl;
            return false;
        }
        
        // Save raw JSON for debugging
        std::string json_path = output_path + ".json";
        std::ofstream json_file(json_path);
        json_file << result_json;
        json_file.close();
        std::cout << "JSON result saved to: " << json_path << std::endl;
        
        return true;
    }

private:
    void render_detections(cv::Mat& image, const std::vector<Detection>& detections) {
        for (const auto& det : detections) {
            // Draw bounding box
            cv::Point pt1(static_cast<int>(det.x1), static_cast<int>(det.y1));
            cv::Point pt2(static_cast<int>(det.x2), static_cast<int>(det.y2));
            cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0), 2);
            
            // Draw label background
            std::string label = det.class_name + " (" + std::to_string(static_cast<int>(det.confidence * 100)) + "%)";
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
            cv::Point label_pt(det.x1, det.y1 - 10);
            if (label_pt.y < 0) label_pt.y = det.y1 + 20;
            
            cv::rectangle(image, 
                         cv::Point(label_pt.x, label_pt.y - text_size.height - 5),
                         cv::Point(label_pt.x + text_size.width, label_pt.y + 5),
                         cv::Scalar(0, 255, 0), -1);
            
            // Draw label text
            cv::putText(image, label, label_pt, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        }
    }
};

int main(int argc, char* argv[]) {
    std::cout << "=== YOLO Image Processor ===" << std::endl;
    
    // Parse command line arguments
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <input_image> <output_image> [engine_path] [output_dir]" << std::endl;
        std::cout << "  input_image: Path to input JPG/PNG image" << std::endl;
        std::cout << "  output_image: Path to save annotated result" << std::endl;
        std::cout << "  engine_path: Path to YOLO engine file (default: ./yolo11n.engine)" << std::endl;
        std::cout << "  output_dir: Output directory for additional files (default: output)" << std::endl;
        return -1;
    }
    
    std::string input_path = argv[1];
    std::string output_path = argv[2];
    std::string engine_path = (argc > 3) ? argv[3] : "./yolo11n.engine";
    std::string output_dir = (argc > 4) ? argv[4] : "output";
    
    // Check if input file exists
    if (!std::filesystem::exists(input_path)) {
        std::cerr << "Input file does not exist: " << input_path << std::endl;
        return -1;
    }
    
    // Initialize YOLO service
    std::cout << "Initializing YOLO service..." << std::endl;
    YoloRgbaService yolo_service;
    
    if (!yolo_service.init(engine_path, 640, 0.25, 0)) {
        std::cerr << "Failed to initialize YOLO model from: " << engine_path << std::endl;
        return -1;
    }
    
    std::cout << "YOLO model initialized successfully!" << std::endl;
    
    // Create image processor
    ImageProcessor processor(yolo_service, output_dir);
    
    // Process image
    bool success = processor.process_image(input_path, output_path);
    
    if (success) {
        std::cout << "Image processing completed successfully!" << std::endl;
        return 0;
    } else {
        std::cerr << "Image processing failed!" << std::endl;
        return -1;
    }
}
