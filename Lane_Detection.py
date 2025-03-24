import cv2
import numpy as np
from picarx import Picarx

px = Picarx()
cap = cv2.VideoCapture(0)  # Open camera

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)  # Adjusted thresholds
    return edges

def detect_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
    return lines if lines is not None else np.array([])

def calculate_lane_center(lines, frame_width):
    left_x, right_x = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
        if -0.5 < slope < 0.5:  # Relaxed slope filtering
            continue
        (left_x if slope < 0 else right_x).append((x1 + x2) // 2)

    if left_x and right_x:
        left_avg = np.mean(left_x)
        right_avg = np.mean(right_x)
        return int((left_avg + right_avg) // 2)
    elif left_x:
        return int(np.mean(left_x))  # If no right lines, use left lines
    elif right_x:
        return int(np.mean(right_x))  # If no left lines, use right lines
    return frame_width // 2  # Default to center if no lines detected

def draw_lines(frame, lines):
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

    # Adjust steering
    steering_adjustment = np.clip((lane_center - frame.shape[1] // 2) * 0.03, -30, 30)  # Adjusted multiplier
    px.set_dir_servo_angle(steering_adjustment)

    # Draw lines on the frame
    draw_lines(frame, lines)
   
    # Draw the lane center point
    cv2.circle(frame, (lane_center, frame.shape[0] // 2), 5, (0, 0, 255), -1)
   
    # Show the camera feed
    cv2.imshow("Camera", frame)

try:
    px.set_cam_tilt_angle(-20)  # Adjust camera tilt angle if necessary
    while True:
        lane_follow()

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            px.foward(20)
            print("Moving forward")
        elif key == ord('s'):
            px.foward(0)
            print("Stopping")
        elif key == ord('q'):
            print("Exiting...")
            break
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    cap.release()
    px.stop()
    cv2.destroyAllWindows()  # Close the camera window properly