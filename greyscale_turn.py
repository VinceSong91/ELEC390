from picarx import Picarx
import time
import cv2
import numpy as np

px = Picarx()
cap = cv2.VideoCapture(0)
NEUTRAL_ANGLE = -13.5
CAMERA_TILT_ANGLE = -20
CAMERA_PAN_ANGLE = -10 # Further adjust to turn the camera more to the left
px.set_cam_tilt_angle(CAMERA_TILT_ANGLE)
px.set_cam_pan_angle(CAMERA_PAN_ANGLE)

WHITE_THRESHOLD = 700  # Adjust this based on your environment

def adjust_direction():
    """Adjust the car's direction based on grayscale sensor values."""
    sensor_values = px.get_grayscale_data()
    left_sensor = sensor_values[0]
    right_sensor = sensor_values[2]



    if left_sensor > 200:
        print("Left sensor detected high value! Turning right.")
        px.set_dir_servo_angle(60)  # Adjust for sharper turns if necessary
    elif right_sensor > 200:
        print("Right sensor detected high value! Turning left.")
        px.set_dir_servo_angle(-80)
    else:
        print("Following straight.")
        px.set_dir_servo_angle(-10)  # Neutral for straight movement

def detect_stop_line():
    """Check for white stop line using grayscale sensors."""
    sensor_values = px.get_grayscale_data()
    print("Grayscale sensor readings:", sensor_values)
    
    # If all sensors detect a high value (likely a white stop line)
    if all(value > WHITE_THRESHOLD for value in sensor_values):
        print("Stop line detected! Stopping the car and waiting for user input.")
        px.stop()  # Stop the car when the stop line is detected
        time.sleep(2)  # Pause for a moment
        wait_for_user_input()  # Wait for user input to continue

def wait_for_user_input():
    """Wait for the user to input a direction for the car."""
    while True:
        print("Please choose a direction:")
        print("1: Turn Left")
        print("2: Turn Right")
        print("3: Move Forward")
        user_input = input("Enter your choice (1/2/3): ").strip()

        if user_input == "1":
            px.turn_signal_left_on()

            print("Turning left.")
            px.forward(5)  # Move forward slowly while turning
            time.sleep(0.40)
            px.set_dir_servo_angle(-30)  # Adjust the angle for left turn
            px.forward(5)  # Move forward slowly while turning
            while px.get_grayscale_data()[0] < 200 and px.get_grayscale_data()[2] < 200:
                continue

            px.turn_signal_left_off()
            main()


        elif user_input == "2":
            px.turn_signal_right_on()

            print("Turning right.")
            px.forward(5)  # Move forward slowly while turning
            time.sleep(0.4)
            px.set_dir_servo_angle(25)  # Adjust the angle for right turn
            px.forward(5)  # Move forward slowly while turning
            while px.get_grayscale_data()[2] < 200:
                continue
            print("Right line detected! Stopping turn.")
            px.turn_signal_right_off()
            main()

        elif user_input == "3":
            print("Moving forward.")
            px.set_dir_servo_angle(-13)  # Neutral for forward movement
            px.forward(10)  # Move forward at a reasonable speed
            break
        else:
            print("Invalid choice, please try again.")


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

    # Crop the frame to only the lower 75% of the image
    height, width = frame.shape[:2]
    lower_75_percent_frame = frame[int(height * 0.25):, :]  # 75% of the lower part

    mask = preprocess_image(lower_75_percent_frame)
    lines = detect_lines(mask)
    lane_center = calculate_lane_center(lines, lower_75_percent_frame.shape[1])

    # Adjust steering
    steering_adjustment = np.clip((lane_center - lower_75_percent_frame.shape[1] // 2) * 0.03, -30, 30)
    final_angle = NEUTRAL_ANGLE + steering_adjustment
    #px.set_dir_servo_angle(final_angle)

    # Draw visualization on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Adjust line drawing to fit in the lower 75% of the frame
            y_offset = int(height * 0.25)  # Offset lines to fit in the lower 75%
            cv2.line(frame, (x1, y1 + y_offset), (x2, y2 + y_offset), (0, 255, 0), 3)

    # Draw the lane center marker on the original frame
    cv2.circle(frame, (lane_center, height // 2), 5, (0, 0, 255), -1)

    cv2.imshow("Lane Detection", frame)



def main():
    try:
        px.forward(5)  # Start moving slowly
        while True:
            # Main loop handles stop line and direction adjustments.
            lane_follow()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            detect_stop_line()  
            adjust_direction()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting program. Stopping the car.")
        px.stop()
    finally:
        cap.release()
        px.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
