from picarx import Picarx
import time
import cv2
import numpy as np
import threading
from queue import Queue
from music import Music

music = Music()


px = Picarx()
cap = cv2.VideoCapture(0)
NEUTRAL_ANGLE = -10.5
CAMERA_TILT_ANGLE = -25
CAMERA_PAN_ANGLE = -15
px.set_cam_tilt_angle(CAMERA_TILT_ANGLE)
px.set_cam_pan_angle(CAMERA_PAN_ANGLE)

WHITE_THRESHOLD = 700
TURN_SPEED = 5
TURN_DELAY = 0.4
DRIVING_SPEED = 10
STOPPED = False
TURNING = False

# Queue for user input commands
command_queue = Queue()

def get_user_input():
    """Run in a separate thread to get user input without blocking main loop"""
    while True:
        print("\nOptions while driving:")
        print("1: Turn Left")
        print("2: Turn Right")
        print("s: Stop/Start")
        print("q: Quit")
        user_input = input("Enter command: ").strip().lower()
        
        if user_input in ['1', '2', 's', 'q']:
            command_queue.put(user_input)
        if user_input == 'q':
            break

def execute_command(command):
    """Execute the user's command"""
    global STOPPED, TURNING
    
    if command == '1':
        TURNING = True
        left_turn()
        TURNING = False
    elif command == '2':
        TURNING = True
        right_turn()
        TURNING = False
    elif command == 's':
        STOPPED = not STOPPED
        if STOPPED:
            px.stop()
            print("Car stopped")
        else:
            px.forward(DRIVING_SPEED)
            print("Car started")
    elif command == 'q':
        px.stop()
        cap.release()
        cv2.destroyAllWindows()
        exit()

def right_turn():
    """Execute a right turn maneuver while maintaining lane following"""
    print("Initiating right turn")
    music.music_play('tokyodrift-368bnuq4-38107.mp3')
    px.turn_signal_right_on()
    time.sleep(1)
    px.forward(10)
    px.set_dir_servo_angle(20)  # Right turn angle
    time.sleep(3.5)  # Turn for 1 second
    px.turn_signal_right_off()
    px.set_dir_servo_angle(NEUTRAL_ANGLE)
    print("Right turn completed")

def left_turn():
    """Execute a left turn maneuver while maintaining lane following"""
    print("Initiating left turn")
    px.turn_signal_left_on()
    time.sleep(1)
    px.forward(10)
    px.set_dir_servo_angle(-27)  # Left turn angle
    time.sleep(3.25)  # Turn for 1 second
    px.turn_signal_left_off()
    px.set_dir_servo_angle(NEUTRAL_ANGLE)
    print("Left turn completed")

def adjust_direction():
    """Adjust the car's direction based on grayscale sensor values."""
    if STOPPED or TURNING:
        return
        
    sensor_values = px.get_grayscale_data()
    left_sensor = sensor_values[0]
    middle_sensor = sensor_values[1]
    right_sensor = sensor_values[2]

    if left_sensor > 200:
        print("Left sensor detected high value! Turning right.")
        px.set_dir_servo_angle(60)
    elif right_sensor > 200:
        print("Right sensor detected high value! Turning left.")
        px.set_dir_servo_angle(-80)
    else:
        px.set_dir_servo_angle(NEUTRAL_ANGLE)

def detect_stop_line():
    """Check for white stop line using grayscale sensors."""
    global STOPPED
    
    if STOPPED or TURNING:
        return
        
    sensor_values = px.get_grayscale_data()
    
    if all(value > WHITE_THRESHOLD for value in sensor_values):
        print("Stop line detected!")
        STOPPED = True
        px.stop()
        time.sleep(2)
        # Wait for user input at stop line
        while STOPPED:
            if not command_queue.empty():
                command = command_queue.get()
                execute_command(command)
            time.sleep(0.1)

# [Keep all your existing lane following functions unchanged]
# remove_white_boards, mask_white_yellow, preprocess_image, 
# detect_lines, calculate_lane_center, draw_lines, lane_follow
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

ULTRASONIC_THRESHOLD = 20
OBSTACLE_DETECTED = False

def check_obstacle():
    """Continuously check for obstacles using the ultrasonic sensor."""
    global STOPPED, OBSTACLE_DETECTED
    while True:
        distance = px.ultrasonic.read()
        if distance <= ULTRASONIC_THRESHOLD and distance > 0:
            print(f"Obstacle detected at {distance} cm! Stopping.")
            OBSTACLE_DETECTED = True
            px.stop()
        elif OBSTACLE_DETECTED and distance > ULTRASONIC_THRESHOLD and distance > 0:
            print("Obstacle cleared. Resuming.")
            OBSTACLE_DETECTED = False
            if not STOPPED:
                time.sleep(1)
                px.forward(DRIVING_SPEED)
        time.sleep(0.1)

def main():
    try:
        # Start user input and obstacle detection threads
        input_thread = threading.Thread(target=get_user_input, daemon=True)
        input_thread.start()

        obstacle_thread = threading.Thread(target=check_obstacle, daemon=True)
        obstacle_thread.start()
        
        px.forward(DRIVING_SPEED)
        
        while True:
            # Check for user commands
            if not command_queue.empty():
                command = command_queue.get()
                execute_command(command)
            
            # Only perform driving operations if not stopped or if no obstacle detected
            if not STOPPED and not OBSTACLE_DETECTED:
                lane_follow()
                detect_stop_line()
                adjust_direction()
            elif OBSTACLE_DETECTED:
                px.stop()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Exiting program")
    finally:
        px.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
