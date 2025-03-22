import time
from picarx import PicarX

import board
import analogio

grayscale_sensor = analogio.AnalogIn(board.A0) # Change this based on how our sensor is connected

WHITE_THRESHOLD = 30000  # Test this value

def read_grayscale():
    return grayscale_sensor.value

def main():
    car = PicarX()

    try:
        while True:
            sensor_value = read_grayscale()
            print("Grayscale value: ", sensor_value) # To test the threshold value

            if sensor_value > WHITE_THRESHOLD:
                car.stop()
                # Figure out condition to move car again (camera check for cars maybe?)
                time.sleep(2)
                car.forward()
            else:
                car.forward()

            time.sleep(0.1)

    except KeyboardInterrupt:
        car.stop()

if __name__ == "__main__":
    main()
