from ultralytics import YOLO
import cv2
import time
from collections import deque

# Configuration
ENGINE = "/root/yolo/yolo11n_fp16.engine"
VIDEO_PATH = "/root/yolo/office_video.mp4"  # Change this to your MP4 file path
IMGSZ = 640
CONF = 0.25
DEVICE = 0

# FPS tracking settings
FPS_WINDOW = 30  # Number of frames to average FPS over

def main():
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(ENGINE)
    print("Model loaded!")
    
    # Open video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {VIDEO_PATH}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    print(f"Video: {total_frames} frames @ {video_fps:.1f} FPS ({duration:.1f}s duration)")
    
    # FPS tracking
    frame_times = deque(maxlen=FPS_WINDOW)
    frame_count = 0
    
    print("Starting detection... Press 'q' to quit, 'space' to pause")
    
    paused = False
    start_time = time.time()
    
    while True:
        if not paused:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            
            # Timing for processing FPS
            process_start = time.time()
            
            # Run YOLO detection
            results = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                device=DEVICE,
                verbose=False
            )
            
            # Get annotated frame
            result = results[0]
            annotated_frame = result.plot()
            
            # Calculate processing time and FPS
            process_time = time.time() - process_start
            frame_times.append(process_time)
            
            processing_fps = 1.0 / process_time if process_time > 0 else 0
            avg_fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0
            
            # Count detections
            num_detections = len(result.boxes) if result.boxes is not None else 0
            
            # Video progress
            progress_percent = (frame_count / total_frames) * 100
            elapsed_time = time.time() - start_time
            
            # Create info text
            info_lines = [
                f"Processing FPS: {processing_fps:.1f} | Avg: {avg_fps:.1f}",
                f"Video FPS: {video_fps:.1f} | Frame: {frame_count}/{total_frames}",
                f"Progress: {progress_percent:.1f}% | Time: {elapsed_time:.1f}s",
                f"Detections: {num_detections}"
            ]
            
            # Add text to frame
            y_offset = 25
            for i, line in enumerate(info_lines):
                y_pos = y_offset + (i * 25)
                # Background rectangle for better readability
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_frame, (5, y_pos - 20), (text_size[0] + 10, y_pos + 5), (0, 0, 0), -1)
                cv2.putText(annotated_frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            frame_count += 1
        
        # Display frame
        cv2.imshow('YOLO MP4 Detection', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar to pause/unpause
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r'):  # 'r' to restart
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            start_time = time.time()
            print("Restarting video")
        elif key == ord('s'):  # 's' to save current frame
            cv2.imwrite(f"frame_{frame_count:06d}.jpg", annotated_frame)
            print(f"Saved frame {frame_count}")
    
    # Cleanup
    total_time = time.time() - start_time
    overall_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\nProcessing complete!")
    print(f"Processed {frame_count} frames in {total_time:.1f}s")
    print(f"Overall processing FPS: {overall_fps:.1f}")
    print(f"Average processing FPS: {avg_fps:.1f}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
