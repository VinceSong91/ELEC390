from picarx import Picarx
import time

px = Picarx()
car = px.forward(50)

def adjust_direction():
    """Adjust the car's direction based on grayscale sensor values."""
    sensor_values = px.get_grayscale_data()

    left_sensor = sensor_values[0]
    right_sensor = sensor_values[2]

    if left_sensor > 200:
        print("Left sensor detected high value! Turning right.")
        px.set_dir_servo_angle(50)  # Adjust the angle as needed
    elif right_sensor > 200:
        print("Right sensor detected high value! Turning left.")
        px.set_dir_servo_angle(-68)  # Adjust the angle as needed
    else:
        px.set_dir_servo_angle(-14)

while True:
    px.forward(10)
    adjust_direction()

    WHITE_THRESHOLD = 700  # Adjust based on testing

def main():
    car = PicarX()

    try:
        while True:
            grayscale_values = car.get_grayscale_data()
            print("Grayscale sensor readings:", grayscale_values)

            # Check if any of the sensors detect a white line
            if any(value > WHITE_THRESHOLD for value in grayscale_values):
                car.stop()
                # Figure out condition to move car again (camera check for cars maybe?)
                time.sleep(2)
                car.forward()
                time.sleep(1.5) # Some time to let the car pass the stop line before starting detection again
            else:
                car.forward()

            time.sleep(0.1)

    except KeyboardInterrupt:
        car.stop()

if __name__ == "__main__":
    main()
