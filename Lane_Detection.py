import cv2
import numpy as np
from picarx import Picarx

class MyHandCodedLaneFollower:
    def __init__(self, car):
        self.car = car
        self.curr_steering_angle = 90  # Straight

    def follow_lane(self, frame):
        lane_lines = self.detect_lane(frame)
        if lane_lines is not None:
            steering_angle = self.compute_steering_angle(frame, lane_lines)
            self.car.set_dir_servo_angle(steering_angle)
        return frame

    def detect_lane(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        mask = np.zeros_like(edges)
        height, width = edges.shape
        region = np.array([[(0, height), (width, height), (width // 2, height // 2)]]);
        cv2.fillPoly(mask, region, 255)

        masked_edges = cv2.bitwise_and(edges, mask)

        # Hough Transform to find lanes
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=200)
        return lines

    def compute_steering_angle(self, frame, lines):
        left_fit = []
        right_fit = []
        height, width, _ = frame.shape
        mid_x = width // 2

        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 0.0001)
            if abs(slope) < 0.5:  # Ignore horizontal lines
                continue
            if slope < 0:  # Left lane
                left_fit.append((slope, y1 - slope * x1))
            else:  # Right lane
                right_fit.append((slope, y1 - slope * x1))

        # Calculate average slopes and intercepts
        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
        else:
            left_fit_average = None
        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
        else:
            right_fit_average = None

        # Determine steering angle
        if left_fit_average is not None and right_fit_average is not None:
            mid_lane_x = (left_fit_average[1] + right_fit_average[1]) / 2
        elif left_fit_average is not None:
            mid_lane_x = left_fit_average[1]
        elif right_fit_average is not None:
            mid_lane_x = right_fit_average[1]
        else:
            return 90  # Keep straight if no lane detected

        steering_angle = 90 + (mid_lane_x - mid_x) / width * 90
        return int(np.clip(steering_angle, 45, 135))
