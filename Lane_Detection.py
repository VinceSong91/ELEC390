import cv2
import numpy as np
from picarx import Picarx
import time

px = Picarx()
cap = cv2.VideoCapture(0)
NEUTRAL_ANGLE = -13.5
CAMERA_TILT_ANGLE = -20
CAMERA_PAN_ANGLE = 0 # Further adjust to turn the camera more to the left
px.set_cam_tilt_angle(CAMERA_TILT_ANGLE)
px.set_cam_pan_angle(CAMERA_PAN_ANGLE)


def preprocess_image(frame):
    """Apply color filtering to isolate white and yellow lanes."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White lane detection
    white_lower = np.array([0, 0, 180])
    white_upper = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    # Yellow lane detection with expanded hue range
    yellow_lower = np.array([15, 100, 100])
    yellow_upper = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    return combined_mask


def detect_lines(mask):
    """Detect lines using Hough Transform."""
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
    return lines if lines is not None else []


def calculate_lane_center(lines, frame_width):
    """Calculate lane center based on detected lines."""
    left_x, right_x = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if -0.5 < slope < 0.5:  # Exclude horizontal lines
            continue
        (left_x if slope < 0 else right_x).append((x1 + x2) // 2)

    if left_x and right_x:
        lane_center = (np.mean(left_x) + np.mean(right_x)) // 2
    elif left_x:
        lane_center = np.mean(left_x)
    elif right_x:
        lane_center = np.mean(right_x)
    else:
        lane_center = frame_width // 2

    return int(lane_center)


def lane_follow():
    ret, frame = cap.read()
    if not ret:
        return

    # Crop the frame to only the lower 65% of the image
    height, width = frame.shape[:2]
    lower_65_percent_frame = frame[int(height * 0.35):, :]  # 65% of the lower part

    mask = preprocess_image(lower_65_percent_frame)
    lines = detect_lines(mask)
    lane_center = calculate_lane_center(lines, lower_65_percent_frame.shape[1])

    # Adjust steering
    steering_adjustment = np.clip((lane_center - lower_65_percent_frame.shape[1] // 2) * 0.03, -30, 30)
    final_angle = NEUTRAL_ANGLE + steering_adjustment
    px.set_dir_servo_angle(final_angle)

    # Draw visualization on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Adjust line drawing to fit in the lower 65% of the frame
            y_offset = int(height * 0.35)  # Offset lines to fit in the lower 65%
            cv2.line(frame, (x1, y1 + y_offset), (x2, y2 + y_offset), (0, 255, 0), 3)

    # Draw the lane center marker on the original frame
    cv2.circle(frame, (lane_center, height // 2), 5, (0, 0, 255), -1)

    cv2.imshow("Lane Detection", frame)


try:
    px.forward(20)
    while True:
        lane_follow()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Exiting...")
finally:
    cap.release()
    px.stop()
    cv2.destroyAllWindows()
