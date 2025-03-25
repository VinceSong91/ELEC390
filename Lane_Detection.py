import cv2
import numpy as np
from picarx import Picarx
import time

# Camera and steering calibration constants
NEUTRAL_ANGLE = -13.5
CAMERA_TILT_ANGLE = -20
CAMERA_PAN_ANGLE = -10  # Adjust to turn the camera more to the left

# Initialize robot and camera
px = Picarx()
cap = cv2.VideoCapture(0)
px.set_cam_tilt_angle(CAMERA_TILT_ANGLE)
px.set_cam_pan_angle(CAMERA_PAN_ANGLE)

def preprocess_image(frame):
    """Apply color filtering and morphological operations to isolate lanes."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White lane detection (solid line)
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 55, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    # Yellow lane detection (dashed line)
    yellow_lower = np.array([15, 80, 100])
    yellow_upper = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Optional: Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    return white_mask, yellow_mask

def detect_lines(mask):
    """Detect lines using Hough Transform on a given mask."""
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=150)
    return lines if lines is not None else []

def calculate_lane_boundaries(lines, frame_width):
    """Separate lines into left and right based on slope and position."""
    left_lines, right_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if abs(slope) < 0.5:  # Exclude near-horizontal lines
            continue
        # Use x position of the line's midpoint to classify
        midpoint = (x1 + x2) / 2.0
        if slope < 0 and midpoint < frame_width / 2:
            left_lines.append(line[0])
        elif slope > 0 and midpoint > frame_width / 2:
            right_lines.append(line[0])
    return left_lines, right_lines

def average_line(lines):
    """Average a list of lines into a single representative line."""
    if len(lines) == 0:
        return None
    try:
        x1_avg = np.mean([line[0] for line in lines])
        y1_avg = np.mean([line[1] for line in lines])
        x2_avg = np.mean([line[2] for line in lines])
        y2_avg = np.mean([line[3] for line in lines])
        return int(x1_avg), int(y1_avg), int(x2_avg), int(y2_avg)
    except IndexError:
        return None

def calculate_lane_center(frame_width, left_line, right_line):
    """Compute lane center as the midpoint between left and right lane boundaries."""
    if left_line is not None and right_line is not None:
        left_x = (left_line[0] + left_line[2]) / 2.0
        right_x = (right_line[0] + right_line[2]) / 2.0
        lane_center = (left_x + right_x) / 2.0
    elif left_line is not None:
        left_x = (left_line[0] + left_line[2]) / 2.0
        lane_center = left_x + frame_width * 0.25
    elif right_line is not None:
        right_x = (right_line[0] + right_line[2]) / 2.0
        lane_center = right_x - frame_width * 0.25
    else:
        lane_center = frame_width / 2.0
    return int(lane_center)

def lane_follow():
    ret, frame = cap.read()
    if not ret:
        return

    height, width = frame.shape[:2]
    roi = frame[int(height * 0.25):, :]
    
    white_mask, yellow_mask = preprocess_image(roi)
    
    white_lines = detect_lines(white_mask)
    yellow_lines = detect_lines(yellow_mask)
    
    white_left, white_right = calculate_lane_boundaries(white_lines, roi.shape[1])
    yellow_left, yellow_right = calculate_lane_boundaries(yellow_lines, roi.shape[1])
    
    left_line = average_line(yellow_left) if yellow_left else average_line(yellow_lines)
    right_line = average_line(white_right) if white_right else average_line(white_lines)
    
    lane_center = calculate_lane_center(roi.shape[1], left_line, right_line)
    
    steering_adjustment = np.clip((lane_center - (roi.shape[1] // 2)) * 0.03, -30, 30)
    final_angle = NEUTRAL_ANGLE + steering_adjustment
    px.set_dir_servo_angle(final_angle)
    
    vis = roi.copy()
    if left_line is not None:
        cv2.line(vis, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 3)
    if right_line is not None:
        cv2.line(vis, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 3)
    cv2.circle(vis, (lane_center, vis.shape[0]//2), 5, (0, 255, 0), -1)
    
    cv2.imshow("Lane Detection", vis)

try:
    px.forward(5)
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