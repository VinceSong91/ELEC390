from picarx import Picarx
import time
import cv2
import numpy as np
import threading
from queue import Queue

px = Picarx()
cap = cv2.VideoCapture(0)
NEUTRAL_ANGLE = -13.5
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
    px.turn_signal_right_on()
    px.set_dir_servo_angle(27)  # Right turn angle
    time.sleep(1)  # Turn for 1 second
    px.turn_signal_right_off()
    px.set_dir_servo_angle(NEUTRAL_ANGLE)
    print("Right turn completed")

def left_turn():
    """Execute a left turn maneuver while maintaining lane following"""
    print("Initiating left turn")
    px.turn_signal_left_on()
    px.set_dir_servo_angle(-25)  # Left turn angle
    time.sleep(1)  # Turn for 1 second
    px.turn_signal_left_off()
    px.set_dir_servo_angle(NEUTRAL_ANGLE)
    print("Left turn completed")

def adjust_direction():
    """Adjust the car's direction based on grayscale sensor values."""
    if STOPPED or TURNING:
        return
        
    sensor_values = px.get_grayscale_data()
    left_sensor = sensor_values[0]
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
    print("Grayscale sensor readings:", sensor_values)
    
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

def main():
    try:
        # Start user input thread
        input_thread = threading.Thread(target=get_user_input, daemon=True)
        input_thread.start()
        
        px.forward(DRIVING_SPEED)
        
        while True:
            # Check for user commands
            if not command_queue.empty():
                command = command_queue.get()
                execute_command(command)
            
            # Only perform driving operations if not stopped
            if not STOPPED and not TURNING:
                lane_follow()
                detect_stop_line()
                adjust_direction()
            
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