from picarx import Picarx
import time
import cv2
import numpy as np

px = Picarx()
cap = cv2.VideoCapture(0)
NEUTRAL_ANGLE = -13.5
CAMERA_TILT_ANGLE = -25
CAMERA_PAN_ANGLE = -15 # Further adjust to turn the camera more to the left
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
            px.set_dir_servo_angle(-25)  # Adjust the angle for left turn
            px.forward(5)  # Move forward slowly while turning
            while True:
                sensor_values = px.get_grayscale_data()
                left_sensor = sensor_values[0]
                right_sensor = sensor_values[2]

                if right_sensor > 200:
                    print("Right lane detected! Stopping turn.")
                    main()
                    break
                elif left_sensor > 200:  # Left sensor detects the line
                    print("Left line detected! Stopping turn.")
                    px.turn_signal_left_off()
                    main()
                    break

        elif user_input == "2":
            px.turn_signal_right_on()

            print("Turning right.")
            px.forward(5)  # Move forward slowly while turning
            time.sleep(0.4)
            px.set_dir_servo_angle(27)  # Adjust the angle for right turn
            px.forward(5)  # Move forward slowly while turning
            while True:
                sensor_values = px.get_grayscale_data()
                right_sensor = sensor_values[2]
                if right_sensor > 200:  # Right sensor detects the line
                    print("Right line detected! Stopping turn.")
                    px.turn_signal_right_off()
                    main()
                    break

        elif user_input == "3":
            print("Moving forward.")
            px.set_dir_servo_angle(-13)  # Neutral for forward movement
            px.forward(10)  # Move forward at a reasonable speed
            break
        else:
            print("Invalid choice, please try again.")


def remove_white_boards(frame):
    height, width = frame.shape[:2]
    # Define margins - adjust these values to cover your white board areas
    left_margin = int(width * 0.019)   # Left 10% of the image
    right_margin = int(width * 0.995)   # Right 10% of the image
   
    # Draw black rectangles on the left and right sides
    cv2.rectangle(frame, (0, 0), (left_margin, height), (0, 0, 0), thickness=-1)
    cv2.rectangle(frame, (right_margin, 0), (width, height), (0, 0, 0), thickness=-1)
   
    return frame

def mask_white_yellow(frame):
    """ Mask white and yellow lane lines in an image. """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([55, 255, 255])
   
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])

    # Create masks
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Combine masks
    combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
    return combined_mask


def preprocess_image(frame):
    """ Convert the frame to edge-detected output for line detection. """
    frame = remove_white_boards(frame)
    masked = mask_white_yellow(frame)  
    blurred = cv2.GaussianBlur(masked, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)  
    return edges


def detect_lines(edges):
    """ Detect lane lines using Hough Transform. """
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
    return lines if lines is not None else np.array([])


def calculate_lane_center(lines, frame_width):
    # left_inside_edges = []
    right_inside_edges = []
   
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate slope and filter out near-horizontal lines
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if abs(slope) < 0.5:
            continue

        if slope > 0:
            inside_edge = min(x1, x2)
            right_inside_edges.append(inside_edge)
       
        # if slope < 0:
        #     # For left lane lines, pick the inside edge (max x value)
        #     inside_edge = max(x1, x2)
        #     left_inside_edges.append(inside_edge)
        # else:
        #     # For right lane lines, pick the inside edge (min x value)
        #     inside_edge = min(x1, x2)
        #     right_inside_edges.append(inside_edge)

   
    # If both left and right inside edges are detected, the lane center is their average.
    # if left_inside_edges and right_inside_edges:
    #     left_avg = np.mean(left_inside_edges)
    #     right_avg = np.mean(right_inside_edges)
    #     lane_center = int((left_avg + right_avg) / 2)
    # elif right_inside_edges:
    #     lane_center = int(np.mean(right_inside_edges)) - 125
    if right_inside_edges:
        lane_center = int(np.mean(right_inside_edges)) - 125
    else:
        lane_center = frame_width // 2  # Fallback to image center if no lane lines found
 
    return lane_center


def draw_lines(frame, lines):
    """ Draw detected lane lines on the frame. """
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)


def lane_follow():
    """ Perform lane following based on camera input. """
    ret, frame = cap.read()
    if not ret or frame is None or frame.shape[0] == 0:
        print("Camera error, skipping frame.")
        return

    frame = remove_white_boards(frame)
    edges = preprocess_image(frame)
    lines = detect_lines(edges)
    lane_center = calculate_lane_center(lines, frame.shape[1])

    # Adjust steering with a controlled response
    steering_adjustment = np.clip((lane_center - frame.shape[1] // 2) * 0.03, -30, 30)
    px.set_dir_servo_angle(steering_adjustment)

    # Draw lane information
    draw_lines(frame, lines)
    cv2.circle(frame, (lane_center, frame.shape[0] // 2), 5, (0, 0, 255), -1)

    # Show the processed camera feed
    cv2.imshow("Camera", frame)

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
