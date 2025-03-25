from picarx import Picarx
import time
import threading

# Global obstacle threshold (in centimeters)
OBSTACLE_THRESHOLD = 10  

px = Picarx()

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
    if all(value > 700 for value in sensor_values):
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
            time.sleep(0.30)
            px.set_dir_servo_angle(-25)  # Adjust the angle for left turn
            px.forward(5)
            time.sleep(1)
            while True:
                sensor_values = px.get_grayscale_data()
                left_sensor = sensor_values[0]
                right_sensor = sensor_values[2]
                if right_sensor > 200:
                    print("Right lane detected! Stopping turn.")
                    px.turn_signal_left_off()
                    return
                elif left_sensor > 200:
                    print("Left line detected! Stopping turn.")
                    px.turn_signal_left_off()
                    return

        elif user_input == "2":
            px.turn_signal_right_on()
            print("Turning right.")
            px.forward(5)
            time.sleep(0.30)
            px.set_dir_servo_angle(17)  # Adjust the angle for right turn
            px.forward(5)
            time.sleep(1)
            while True:
                sensor_values = px.get_grayscale_data()
                right_sensor = sensor_values[2]
                if right_sensor > 200:
                    print("Right line detected! Stopping turn.")
                    px.turn_signal_right_off()
                    return
        elif user_input == "3":
            print("Moving forward.")
            px.set_dir_servo_angle(-13)  # Neutral for forward movement
            px.forward(10)  # Move forward at a reasonable speed
            return
        else:
            print("Invalid choice, please try again.")

def ultrasonic_monitor():
    """Background thread function to continuously check for obstacles."""
    while True:
        distance = px.get_distance()
        if distance is not None and distance < OBSTACLE_THRESHOLD:
            print("Ultrasonic: Obstacle detected at", distance, "cm. Stopping car.")
            px.stop()
        time.sleep(0.1)

def main():
    # Start the ultrasonic monitor in a background thread.
    monitor_thread = threading.Thread(target=ultrasonic_monitor, daemon=True)
    monitor_thread.start()
    
    try:
        px.forward(5)  # Start moving slowly
        while True:
            # Main loop handles stop line and direction adjustments.
            detect_stop_line()  
            adjust_direction()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting program. Stopping the car.")
        px.stop()

if __name__ == "__main__":
    main()
