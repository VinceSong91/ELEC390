import cv2
import numpy as np
from picarx import Picarx
import time

px = Picarx()
cap = cv2.VideoCapture(0)
NEUTRAL_ANGLE = -13
CAMERA_TILT_ANGLE = -30

def mask_lanes(frame):
    """Mask white lanes and yellow dotted lines using HSV."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White lane masking
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Yellow lane masking
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine both masks
    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
    result = cv2.bitwise_and(frame, frame, mask=lane_mask)
    return result

def preprocess_image(frame):
    """Apply preprocessing: mask lanes, grayscale, blur, and Canny."""
    masked_frame = mask_lanes(frame)
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def detect_lines(edges):
    """Detect lines using Hough Transform."""
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
    return lines if lines is not None else []

def calculate_lane_center(lines, frame_width):
    """Calculate lane center based on detected lines."""
    left_x, right_x = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if -0.5 < slope < 0.5:
            continue
        if slope < 0:
            left_x.append((x1 + x2) // 2)
        else:
            right_x.append((x1 + x2) // 2)

    if left_x and right_x:
        lane_center = (np.mean(left_x) + np.mean(right_x)) // 2
    elif left_x:
        lane_center = np.mean(left_x)
    elif right_x:
        lane_center = np.mean(right_x)
    else:
        lane_center = frame_width // 2

    return int(lane_center)

def draw_lines(frame, lines):
    """Draw detected lane lines."""
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

def lane_follow():
    ret, frame = cap.read()
    if not ret:
        return

    edges = preprocess_image(frame)
    lines = detect_lines(edges)
    lane_center = calculate_lane_center(lines, frame.shape[1])

    # Adjust steering based on lane center
    steering_adjustment = np.clip((lane_center - frame.shape[1] // 2) * 0.03, -30, 30)
    final_angle = NEUTRAL_ANGLE + steering_adjustment
    px.set_dir_servo_angle(final_angle)

    # Visualization
    draw_lines(frame, lines)
    cv2.circle(frame, (lane_center, frame.shape[0] // 2), 5, (0, 0, 255), -1)
    cv2.imshow("Lane Detection", frame)

try:
    px.set_cam_tilt_angle(CAMERA_TILT_ANGLE)
    px.forward(20)

    while True:
        lane_follow()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    cap.release()
    px.stop()
    cv2.destroyAllWindows()
