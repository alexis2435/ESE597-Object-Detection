import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


from ultralytics import YOLO
import cv2
import time
from collections import deque
import numpy as np


class TemporalSmoothing:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.detection_history = {}

    def smooth_detections(self, current_boxes, track_ids):
        if len(current_boxes) == 0:
            return current_boxes

        smoothed_boxes = []

        for i, (box, track_id) in enumerate(zip(current_boxes, track_ids)):
            if track_id not in self.detection_history:
                self.detection_history[track_id] = deque(maxlen=self.window_size)

            self.detection_history[track_id].append(box[:4])

            if len(self.detection_history[track_id]) > 0:
                history = np.array(self.detection_history[track_id])
                smoothed_coords = np.mean(history, axis=0)
                smoothed_box = list(smoothed_coords) + [box[4], box[5]]
                smoothed_boxes.append(smoothed_box)
            else:
                smoothed_boxes.append(box)

        return smoothed_boxes


def detect_from_camera(model_path, camera_id=0, use_smoothing=True):
    print("\n" + "=" * 60)
    print("Camera Real-Time Detection Mode")
    print("=" * 60)

    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}")
        return

    print(f"Camera opened successfully")
    print("\nPress 'q' to quit, 's' to save screenshot")
    print("-" * 60)

    smoother = TemporalSmoothing(window_size=5) if use_smoothing else None
    fps_queue = deque(maxlen=30)
    frame_count = 0

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Cannot read camera frame")
            break

        results = model.track(frame, persist=True, verbose=False)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.data.cpu().numpy()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = list(range(len(boxes)))

            if use_smoothing and smoother:
                boxes = smoother.smooth_detections(boxes, track_ids)

            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame

        elapsed = time.time() - start_time
        fps = 1 / elapsed if elapsed > 0 else 0
        fps_queue.append(fps)
        avg_fps = np.mean(fps_queue)

        info_text = f"FPS: {avg_fps:.1f} | Frame: {frame_count}"
        if use_smoothing:
            info_text += " | Smoothing: ON"

        cv2.putText(annotated_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'screenshot_{frame_count}.jpg'
            cv2.imwrite(filename, annotated_frame)
            print(f"Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"Detection completed! Average FPS: {avg_fps:.2f}")
    print("=" * 60)


def detect_from_video(model_path, video_path, use_smoothing=True, save_output=False):
    print("\n" + "=" * 60)
    print("Video File Detection Mode")
    print("=" * 60)

    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return

    fps_original = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height}, {fps_original} FPS, {total_frames} frames")
    print("\nPress 'q' to quit, 'p' to pause/resume")
    print("-" * 60)

    out = None
    if save_output:
        output_path = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps_original, (width, height))
        print(f"Output will be saved to: {output_path}")

    smoother = TemporalSmoothing(window_size=5) if use_smoothing else None
    fps_queue = deque(maxlen=30)
    frame_count = 0
    paused = False

    while True:
        if not paused:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("\nVideo playback completed")
                break

            results = model.track(frame, persist=True, verbose=False)

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.data.cpu().numpy()

                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = list(range(len(boxes)))

                if use_smoothing and smoother:
                    boxes = smoother.smooth_detections(boxes, track_ids)

                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame

            elapsed = time.time() - start_time
            fps = 1 / elapsed if elapsed > 0 else 0
            fps_queue.append(fps)
            avg_fps = np.mean(fps_queue)

            progress = (frame_count / total_frames) * 100
            info_text = f"FPS: {avg_fps:.1f} | Progress: {progress:.1f}%"
            if use_smoothing:
                info_text += " | Smoothing: ON"

            cv2.putText(annotated_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if save_output and out:
                out.write(annotated_frame)

            cv2.imshow('YOLOv8 Video Detection', annotated_frame)

            frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"Detection completed! Average FPS: {np.mean(fps_queue):.2f}")
    if save_output:
        print(f"Output saved: {output_path}")
    print("=" * 60)


def main():
    model_path = r"D:\code\Pyhton\ESE597-Object-Detection-main\runs\detect\train_full_coco\weights\best.pt"

    print("\n" + "=" * 60)
    print("YOLOv8 Video Stream Detection System")
    print("=" * 60)
    print("\nSelect detection mode:")
    print("1. Camera real-time detection")
    print("2. Video file detection")
    print("3. Online video detection (test)")

    choice = input("\nEnter option (1/2/3): ").strip()

    if choice == '1':
        detect_from_camera(model_path, camera_id=0, use_smoothing=True)

    elif choice == '2':
        video_path = input("Enter video file path: ").strip()
        save = input("Save output video? (y/n): ").strip().lower() == 'y'
        detect_from_video(model_path, video_path, use_smoothing=True, save_output=save)

    elif choice == '3':
        test_video = 'https://ultralytics.com/images/bus.jpg'
        print("Using test image (can replace with video URL)")
        detect_from_video(model_path, test_video, use_smoothing=True, save_output=False)

    else:
        print("Invalid option")


if __name__ == '__main__':
    main()