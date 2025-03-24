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
        print("Stop line detected! Stopping the car.")
        px.stop()
        time.sleep(2)  # Pause for 2 seconds
        wait_for_user_input()  # Wait for user input after detecting stop line

def wait_for_user_input():
    """Wait for the user to input a direction for the car."""
    while True:
        print("Please choose a direction:")
        print("1: Turn Left")
        print("2: Turn Right")
        user_input = input("Enter your choice (1/2): ").strip()

        if user_input == "1":
            print("Turning left for 3 seconds.")
            px.set_dir_servo_angle(-76)  # Adjust the angle for left turn
            px.forward(10)  # Move forward after turning
            time.sleep(3)  # Turn for 3 seconds
            px.stop()
            break
        elif user_input == "2":
            print("Turning right for 3 seconds.")
            px.set_dir_servo_angle(50)  # Adjust the angle for right turn
            px.forward(10)  # Move forward after turning
            time.sleep(3)  # Turn for 3 seconds
            px.stop()
            break
        else:
            print("Invalid choice, please try again.")

def main():
    try:
        px.forward(10)  # Start moving slowly
        while True:
            detect_stop_line()
            adjust_direction()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting program. Stopping the car.")
        px.stop()

if __name__ == "__main__":
    main()
