import cv2
import numpy as np
from picarx import Picarx
import time

# Camera and steering calibration constants
NEUTRAL_ANGLE = -13.5
CAMERA_TILT_ANGLE = -20
CAMERA_PAN_ANGLE = -10
WHITE_THRESHOLD = 700

# Initialize robot and camera
px = Picarx()
cap = cv2.VideoCapture(0)
px.set_cam_tilt_angle(CAMERA_TILT_ANGLE)
px.set_cam_pan_angle(CAMERA_PAN_ANGLE)

def preprocess_image(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 55, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    yellow_lower = np.array([15, 80, 100])
    yellow_upper = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    return white_mask, yellow_mask

def detect_lines(mask):
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=150)
    return lines if lines is not None else []

def average_line(lines):
    if len(lines) == 0:
        return None
    x1_avg = np.mean([line[0] for line in lines])
    y1_avg = np.mean([line[1] for line in lines])
    x2_avg = np.mean([line[2] for line in lines])
    y2_avg = np.mean([line[3] for line in lines])
    return int(x1_avg), int(y1_avg), int(x2_avg), int(y2_avg)

def detect_stop_line(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Detect edges using Canny
    edges = cv2.Canny(thresh, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # If we find lines, return True (stop line detected)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y2 > 0.85 * frame.shape[0]:  # If the line is near the bottom of the frame (indicating a stop line)
                return True
    return False

def lane_follow():
    ret, frame = cap.read()
    if not ret:
        return

    height, width = frame.shape[:2]
    roi = frame[int(height * 0.25):, :]

    white_mask, yellow_mask = preprocess_image(roi)
    white_lines = detect_lines(white_mask)
    yellow_lines = detect_lines(yellow_mask)

    left_line = average_line(yellow_lines)
    right_line = average_line(white_lines)

    if left_line and right_line:
        lane_center = (left_line[0] + right_line[2]) // 2
    elif left_line:
        lane_center = left_line[0] + (width // 4)
    elif right_line:
        lane_center = right_line[2] - (width // 4)
    else:
        lane_center = width // 2

    steering_adjustment = np.clip((lane_center - (width // 2)) * 0.03, -30, 30)
    final_angle = NEUTRAL_ANGLE + steering_adjustment
    px.set_dir_servo_angle(final_angle)

def adjust_direction_with_grayscale():
    sensor_values = px.get_grayscale_data()
    left_sensor = sensor_values[0]
    right_sensor = sensor_values[2]

    if left_sensor > WHITE_THRESHOLD:
        print("Left sensor detected high value! Turning right.")
        px.set_dir_servo_angle(60)
    elif right_sensor > WHITE_THRESHOLD:
        print("Right sensor detected high value! Turning left.")
        px.set_dir_servo_angle(-80)
    else:
        lane_follow()

def wait_for_user_input():
    print("Stop line detected. Press Enter to continue.")
    while True:
        if cv2.waitKey(1) & 0xFF == 13:  # Wait for Enter key
            print("Resuming movement...")
            break

def main():
    try:
        px.forward(10)
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Check for stop line
            if detect_stop_line(frame):
                wait_for_user_input()

            adjust_direction_with_grayscale()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting program. Stopping the car.")
        px.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
