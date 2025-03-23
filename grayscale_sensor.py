import time
from picarx import PicarX

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
