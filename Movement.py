import logging
import cv2
import time
from Lane_Detection import MyHandCodedLaneFollower
from picarx import Picarx

_SHOW_IMAGE = True

class DeepPiCar(object):

    __INITIAL_SPEED = 0
    __SCREEN_WIDTH = 320
    __SCREEN_HEIGHT = 240

    def __init__(self):
        """ Init camera and wheels"""
        logging.info('Creating a DeepPiCar...')

        # Setup for Picarx
        self.px = Picarx()
        
        logging.debug('Set up camera')
        self.camera = cv2.VideoCapture(0)
        self.camera.set(3, self.__SCREEN_WIDTH)
        self.camera.set(4, self.__SCREEN_HEIGHT)

        if not self.camera.isOpened():
            raise RuntimeError("Error: Failed to initialize the camera")

        # Camera servos (pan and tilt)
        self.px.set_cam_pan_angle(0)
        self.px.set_cam_tilt_angle(0)

        logging.debug('Set up wheels')
        self.px.forward(0)  # Stop initially
        
        self.lane_follower = MyHandCodedLaneFollower(self)

    def __enter__(self):
        return self

    def __exit__(self, _type, value, traceback):
        self.cleanup()

    def cleanup(self):
        logging.info('Stopping the car and resetting hardware.')
        self.px.forward(0)
        self.px.set_dir_servo_angle(0)
        if self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()

    def drive(self, speed=40):
        """ Drive continuously using lane detection """
        logging.info(f'Starting to drive at speed {speed}...')
        self.px.forward(speed)

        while True:
            ret, frame = self.camera.read()
            if not ret:
                logging.error("Failed to capture image")
                break

            frame = self.follow_lane(frame)
            show_image('Lane Lines', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Stopping car...")
                self.cleanup()
                break

    def follow_lane(self, image):
        return self.lane_follower.follow_lane(image)


############################
# Utility Functions
############################
def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        try:
            cv2.imshow(title, frame)
        except cv2.error as e:
            print(f"OpenCV Error: {e}")

def main():
    with DeepPiCar() as car:
        try:
            print("Driving continuously. Press 'q' to stop.")
            car.drive(40)
        except KeyboardInterrupt:
            print("Stopping the car.")
            car.cleanup()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s:%(asctime)s: %(message)s')
    main()
