import logging
import cv2
import datetime
import time
from Lane_Detection import Lane_Detection
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
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(3, self.__SCREEN_WIDTH)
        self.camera.set(4, self.__SCREEN_HEIGHT)

        # Camera servos (pan and tilt)
        self.px.set_camera_servo1_angle(90)  # Starting position for pan
        self.px.set_camera_servo2_angle(90)  # Starting position for tilt

        logging.debug('Set up wheels')
        # Back wheels and speed
        self.px.forward(0)  # Stop initially
        
        # Front wheels (steering control)
        self.px.set_dir_servo_angle(90)  # Neutral steering position (straight)

        self.lane_follower = Lane_Detection(self)

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        datestr = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.video_orig = self.create_video_recorder('../data/tmp/car_video%s.avi' % datestr)
        self.video_lane = self.create_video_recorder('../data/tmp/car_video_lane%s.avi' % datestr)
        self.video_objs = self.create_video_recorder('../data/tmp/car_video_objs%s.avi' % datestr)

        logging.info('Created a DeepPiCar')

    def create_video_recorder(self, path):
        return cv2.VideoWriter(path, self.fourcc, 20.0, (self.__SCREEN_WIDTH, self.__SCREEN_HEIGHT))

    def __enter__(self):
        """ Entering a with statement """
        return self

    def __exit__(self, _type, value, traceback):
        """ Exit a with statement"""
        if traceback is not None:
            logging.error('Exiting with statement with exception %s' % traceback)
        self.cleanup()

    def cleanup(self):
        """ Reset the hardware"""
        logging.info('Stopping the car, resetting hardware.')
        self.px.forward(0)
        self.px.set_dir_servo_angle(90)  # Reset to neutral
        self.camera.release()
        self.video_orig.release()
        self.video_lane.release()
        self.video_objs.release()
        cv2.destroyAllWindows()

    def drive(self, speed=__INITIAL_SPEED):
        """ Main entry point of the car, and put it in drive mode
        Keyword arguments:
        speed -- speed of back wheel, range is 0 (stop) - 100 (fastest)
        """
        logging.info('Starting to drive at speed %s...' % speed)
        self.px.forward(speed)
        
        # Simulate pan and tilt movement
        self.simulate_camera_movement()

        # Simulate steering movement
        self.simulate_steering_movement()

        while self.camera.isOpened():
            _, image_lane = self.camera.read()
            image_objs = image_lane.copy()
            self.video_orig.write(image_lane)

            image_lane = self.follow_lane(image_lane)
            self.video_lane.write(image_lane)
            show_image('Lane Lines', image_lane)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cleanup()
                break

    def follow_lane(self, image):
        image = self.lane_follower.follow_lane(image)
        return image

    def simulate_camera_movement(self):
        """ Simulate the camera pan and tilt servo movements """
        for angle in range(0, 35):
            self.px.set_camera_servo1_angle(90 + angle)
            time.sleep(0.01)
        for angle in range(35, -35, -1):
            self.px.set_camera_servo1_angle(90 + angle)
            time.sleep(0.01)
        for angle in range(-35, 0):
            self.px.set_camera_servo1_angle(90 + angle)
            time.sleep(0.01)
        
        for angle in range(0, 35):
            self.px.set_camera_servo2_angle(90 + angle)
            time.sleep(0.01)
        for angle in range(35, -35, -1):
            self.px.set_camera_servo2_angle(90 + angle)
            time.sleep(0.01)
        for angle in range(-35, 0):
            self.px.set_camera_servo2_angle(90 + angle)
            time.sleep(0.01)

    def simulate_steering_movement(self):
        """ Simulate steering wheel servo movement """
        for angle in range(0, 35):
            self.px.set_dir_servo_angle(90 + angle)
            time.sleep(0.01)
        for angle in range(35, -35, -1):
            self.px.set_dir_servo_angle(90 + angle)
            time.sleep(0.01)
        for angle in range(-35, 0):
            self.px.set_dir_servo_angle(90 + angle)
            time.sleep(0.01)


############################
# Utility Functions
############################
def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)


def main():
    with DeepPiCar() as car:
        car.drive(40)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s:%(asctime)s: %(message)s')
    main()
