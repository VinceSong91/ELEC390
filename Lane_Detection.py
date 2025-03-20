import numpy as np
import cv2
from picarx import Picarx

class LaneDetection:
    def __init__(self):
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(3, 320)
        self.camera.set(4, 240)
        self.px = Picarx()

    def region_selection(self, image):
        mask = np.zeros_like(image)
        ignore_mask_color = 255
        rows, cols = image.shape[:2]
        vertices = np.array([[
            [cols * 0.1, rows * 0.95],
            [cols * 0.4, rows * 0.6],
            [cols * 0.6, rows * 0.6],
            [cols * 0.9, rows * 0.95]
        ]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        return cv2.bitwise_and(image, mask)

    def hough_transform(self, image):
        return cv2.HoughLinesP(image, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=500)

    def follow_lane(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        region = self.region_selection(edges)
        lines = self.hough_transform(region)

        if lines is not None:
            left_lane, right_lane = self.average_slope_intercept(lines)
            self.steer_car(left_lane, right_lane)
            return self.draw_lane_lines(image, left_lane, right_lane)
        return image

    def average_slope_intercept(self, lines):
        left_lines = []
        right_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            intercept = y1 - slope * x1
            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))
        
        left_lane = np.mean(left_lines, axis=0) if left_lines else None
        right_lane = np.mean(right_lines, axis=0) if right_lines else None
        return left_lane, right_lane

    def draw_lane_lines(self, image, left_lane, right_lane):
        line_image = np.zeros_like(image)
        for lane in [left_lane, right_lane]:
            if lane is not None:
                slope, intercept = lane
                y1 = image.shape[0]
                y2 = int(y1 * 0.6)
                x1 = int((y1 - intercept) / slope)
                x2 = int((y2 - intercept) / slope)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return cv2.addWeighted(image, 0.8, line_image, 1, 0)

    def steer_car(self, left_lane, right_lane):
        if left_lane is not None and right_lane is not None:
            avg_slope = (left_lane[0] + right_lane[0]) / 2
        elif left_lane is not None:
            avg_slope = left_lane[0]
        elif right_lane is not None:
            avg_slope = right_lane[0]
        else:
            avg_slope = 0

        steering_angle = int(90 - np.degrees(np.arctan(avg_slope)))
        steering_angle = np.clip(steering_angle, 45, 135)
        self.px.set_dir_servo_angle(steering_angle)

    def run(self):
        print("Press 'q' to stop.")
        self.px.forward(30)

        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            processed_frame = self.follow_lane(frame)
            cv2.imshow('Lane Detection', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.px.forward(0)
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    LaneDetection().run()