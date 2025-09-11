from ultralytics import YOLO
import cv2
import time
from collections import deque
import os

# Configuration
ENGINE = "/root/yolo/yolo11n_fp16.engine"
VIDEO_PATH = "/root/yolo/office_video.mp4"  # Change this to your MP4 file path
OUTPUT_PATH = "/root/yolo/yolo_output.mp4"  # Annotated output when headless
IMGSZ = 640
CONF = 0.25
DEVICE = 0

# FPS tracking settings
FPS_WINDOW = 30  # Number of frames to average FPS over

def try_init_window(window_name="YOLO MP4 Detection"):
    """Return True if we can create a GUI window, else False."""
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)
        return True
    except cv2.error:
        return False

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
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps and video_fps > 0 else 0
    
    print(f"Video: {total_frames} frames @ {video_fps:.1f} FPS ({duration:.1f}s duration)")
    
    # Decide display mode
    window_name = 'YOLO MP4 Detection'
    can_show = try_init_window(window_name)
    if not can_show:
        # Prepare VideoWriter for headless mode
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' if available in your build
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, video_fps if video_fps > 0 else 30.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output file for writing: {OUTPUT_PATH}")
        print(f"No GUI backend detected. Saving annotated video to: {OUTPUT_PATH}")
    else:
        writer = None
        print("GUI detected. Press 'q' to quit, 'space' to pause, 'r' to restart, 's' to save a frame.")
    
    # FPS tracking
    frame_times = deque(maxlen=FPS_WINDOW)
    frame_count = 0
    paused = False
    start_time = time.time()
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            
            process_start = time.time()
            results = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                device=DEVICE,
                verbose=False
            )
            result = results[0]
            annotated_frame = result.plot()

            # Overlay info
            process_time = time.time() - process_start
            frame_times.append(process_time)
            processing_fps = 1.0 / process_time if process_time > 0 else 0
            avg_fps = len(frame_times) / sum(frame_times) if frame_times and sum(frame_times) > 0 else 0
            num_detections = len(result.boxes) if result.boxes is not None else 0
            progress_percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            elapsed_time = time.time() - start_time

            info_lines = [
                f"Processing FPS: {processing_fps:.1f} | Avg: {avg_fps:.1f}",
                f"Video FPS: {video_fps:.1f} | Frame: {frame_count}/{total_frames}",
                f"Progress: {progress_percent:.1f}% | Time: {elapsed_time:.1f}s",
                f"Detections: {num_detections}"
            ]
            y_offset = 25
            for i, line in enumerate(info_lines):
                y_pos = y_offset + (i * 25)
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_frame, (5, y_pos - 20), (text_size[0] + 10, y_pos + 5), (0, 0, 0), -1)
                cv2.putText(annotated_frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Write or show
            if writer is not None:
                writer.write(annotated_frame)
            else:
                cv2.imshow(window_name, annotated_frame)

            frame_count += 1

        # Handle keys only when a window exists
        if writer is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                start_time = time.time()
                print("Restarting video")
            elif key == ord('s'):
                fname = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(fname, annotated_frame)
                print(f"Saved {fname}")
        else:
            # In headless mode, just continue; optionally throttle a bit if needed
            pass

    # Cleanup
    total_time = time.time() - start_time
    overall_fps = frame_count / total_time if total_time > 0 else 0
    print("\nProcessing complete!")
    print(f"Processed {frame_count} frames in {total_time:.1f}s")
    print(f"Overall processing FPS: {overall_fps:.1f}")
    if frame_times and sum(frame_times) > 0:
        print(f"Average processing FPS: {len(frame_times) / sum(frame_times):.1f}")

    cap.release()
    if writer is not None:
        writer.release()
    if can_show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
