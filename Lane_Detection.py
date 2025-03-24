import cv2
import numpy as np
from picarx import Picarx
import time

px = Picarx()
cap = cv2.VideoCapture(0)  # Initialize the camera

def preprocess_image(frame):
    """Convert to grayscale, apply blur, and detect edges using Canny."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Prevent division by zero
        if -0.5 < slope < 0.5:  # Filter nearly horizontal lines
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
    px.set_dir_servo_angle(steering_adjustment)

    # Visualization
    draw_lines(frame, lines)
    cv2.circle(frame, (lane_center, frame.shape[0] // 2), 5, (0, 0, 255), -1)
    cv2.imshow("Lane Detection", frame)

try:
    px.set_cam_tilt_angle(-20)  # Adjust the camera angle if needed
    px.forward(20)  # Start moving forward

    while True:
        lane_follow()

        # Stop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    cap.release()
    px.stop()
    cv2.destroyAllWindows()
