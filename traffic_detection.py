# traffic_detection.py
import cv2
from ultralytics import YOLO
import numpy as np
import os
import logging
import time

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(input_path, output_path):
    start_time = time.time()
    logger.info(f"Starting to process video: {input_path}")
    
    try:
        model = YOLO('models/yolov8n.pt')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {input_path}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    if not out.isOpened():
        logger.error(f"Cannot create output video file: {output_path}")
        cap.release()
        return

    class_names = model.names
    traffic_light_status = -1

    def preprocess_frame(frame):
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=5)
        return frame

    def detect_traffic_light_color(roi):
        roi = preprocess_frame(roi)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Điều chỉnh phạm vi màu để tăng độ nhạy
        red_lower1 = np.array([0, 50, 50])  # Giảm saturation để nhận diện đỏ tốt hơn
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        green_lower = np.array([35, 40, 40])  # Điều chỉnh để nhận diện xanh
        green_upper = np.array([85, 255, 255])
        yellow_lower = np.array([15, 50, 50])
        yellow_upper = np.array([40, 255, 255])

        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_green = cv2.inRange(hsv, green_lower, green_upper)
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)
        yellow_pixels = cv2.countNonZero(mask_yellow)

        total_pixels = roi.shape[0] * roi.shape[1]
        red_ratio = red_pixels / total_pixels
        green_ratio = green_pixels / total_pixels
        yellow_ratio = yellow_pixels / total_pixels

        # Thêm log để kiểm tra tỷ lệ màu
        logger.info(f"Color ratios - Red: {red_ratio:.4f}, Green: {green_ratio:.4f}, Yellow: {yellow_ratio:.4f}")

        # Giảm ngưỡng để tăng độ nhạy nhận diện
        if red_ratio > 0.03:
            return 2  # Đèn đỏ
        elif green_ratio > 0.03:
            return 0  # Đèn xanh
        elif yellow_ratio > 0.03:
            return 1  # Đèn vàng
        
        return -1  # Không xác định được màu

    frame_count = 0
    traffic_light_detected = False  # Biến để kiểm tra xem có phát hiện đèn không

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:  # Bỏ qua mỗi 2 frame để tăng tốc
            out.write(frame)
            continue

        frame = preprocess_frame(frame)
        results = model(frame, imgsz=1280, device='cpu')  # Tăng imgsz lên 1280 để nhận diện đèn tốt hơn

        stop_line = int(frame.shape[0] * 0.5)  # Đặt vạch dừng ở giữa khung hình
        cv2.line(frame, (0, stop_line), (frame.shape[1], stop_line), (255, 255, 255), 2)

        for result in results:
            boxes = result.boxes.xyxy
            confidences = result.boxes.conf
            classes = result.boxes.cls

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                confidence = confidences[i].item()
                class_id = int(classes[i].item())
                label = class_names[class_id]

                if label in ['car', 'motorcycle', 'truck', 'bus']:
                    color = (0, 255, 0)  # Màu mặc định là xanh (không vi phạm)
                    width = x2 - x1
                    height = y2 - y1
                    is_vertical = height > width

                    # Kiểm tra vi phạm khi đèn đỏ
                    if is_vertical:
                        if traffic_light_status == 2:  # Đèn đỏ
                            if y1 < stop_line:  # Xe vượt qua vạch dừng
                                color = (0, 0, 255)  # Màu đỏ cho vi phạm
                                cv2.putText(frame, "VI PHAM", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                                logger.info(f"Violation detected: {label} at position ({x1}, {y1})")
                            else:
                                color = (0, 255, 0)  # Không vi phạm
                        elif traffic_light_status == 0:  # Đèn xanh
                            color = (0, 255, 0)  # Không vi phạm
                        elif traffic_light_status == 1:  # Đèn vàng
                            color = (0, 255, 0)  # Tạm thời không coi là vi phạm
                        elif traffic_light_status == -1:  # Không xác định được đèn
                            color = (0, 255, 0)  # Không vi phạm
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                elif label == 'person':
                    color = (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                elif label == 'traffic light':
                    traffic_light_detected = True
                    logger.info(f"Traffic light detected at position ({x1}, {y1}, {x2}, {y2}) with confidence {confidence:.2f}")
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        traffic_light_status = detect_traffic_light_color(roi)
                        logger.info(f"Traffic light status: {traffic_light_status}")
                    
                    color = (255, 255, 0)  # Màu mặc định cho đèn giao thông
                    status_text = "Not a Traffic Light"
                    if traffic_light_status == 0:
                        status_text = "Green"
                        color = (0, 255, 0)
                    elif traffic_light_status == 1:
                        status_text = "Yellow"
                        color = (0, 255, 255)
                    elif traffic_light_status == 2:
                        status_text = "Red"
                        color = (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Traffic Light: {status_text}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    if not traffic_light_detected:
        logger.warning("No traffic light detected in the video.")
    logger.info(f"Finished processing video. Output saved to: {output_path}")
    end_time = time.time()
    logger.info(f"Processing time: {end_time - start_time} seconds")
    return output_path

if __name__ == "__main__":
    input_video = "uploads"
    output_video = "output"
    process_video(input_video, output_video)