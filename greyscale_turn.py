from picarx import Picarx
import time

px = Picarx()
WHITE_THRESHOLD = 700  # Adjust this based on your environment

def adjust_direction():
    """Adjust the car's direction based on grayscale sensor values."""
    sensor_values = px.get_grayscale_data()
    left_sensor = sensor_values[0]
    right_sensor = sensor_values[2]

    if left_sensor > 200:
        print("Left sensor detected high value! Turning right.")
        px.set_dir_servo_angle(50)  # Adjust for sharper turns if necessary
    elif right_sensor > 200:
        print("Right sensor detected high value! Turning left.")
        px.set_dir_servo_angle(-76)
    else:
        print("Following straight.")
        px.set_dir_servo_angle(-13)  # Neutral for straight movement

def detect_stop_line():
    """Check for white stop line using grayscale sensors."""
    sensor_values = px.get_grayscale_data()
    print("Grayscale sensor readings:", sensor_values)
    
    # If all sensors detect a high value (likely a white stop line)
    if all(value > WHITE_THRESHOLD for value in sensor_values):
        print("Stop line detected!")
        return True
    return False

def wait_for_user_input():
    """Wait for user input to determine the next action."""
    while True:
        user_input = input("Enter 'l' to turn left, 'r' to turn right, or 'f' to go forward: ").lower()
        if user_input in ['l', 'r', 'f']:
            return user_input
        else:
            print("Invalid input. Please enter 'l', 'r', or 'f'.")

def main():
    try:
        px.forward(10)  # Start moving slowly
        while True:
            if detect_stop_line():
                action = wait_for_user_input()
                if action == 'l':
                    print("Turning left for 3 seconds.")
                    px.set_dir_servo_angle(50)
                    time.sleep(3)
                elif action == 'r':
                    print("Turning right for 3 seconds.")
                    px.set_dir_servo_angle(-76)
                    time.sleep(3)
                elif action == 'f':
                    print("Continuing forward.")
                    px.forward(10)
                # After action, continue straight
                px.set_dir_servo_angle(-13)  # Neutral for straight movement
                time.sleep(0.1)
            else:
                adjust_direction()
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting program. Stopping the car.")
        px.stop()

if __name__ == "__main__":
    main()
