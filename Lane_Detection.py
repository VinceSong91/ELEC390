import cv2
import numpy as np
from picarx import Picarx
import time

class LaneDetection:
    def __init__(self):
        self.camera = cv2.VideoCapture(-1)
        if not self.camera.isOpened():
            raise Exception("Failed to open camera")
        self.camera.set(3, 640)
        self.camera.set(4, 480)
        
        # Initialize camera tilt using servo
        self.px = Picarx()
        self.px.set_cam_tilt_angle(-20)  # Tilt camera downwards

    def perspective_transform(self, image):
        height, width = image.shape[:2]
        src = np.float32([
            [width * 0.2, height * 0.8],
            [width * 0.8, height * 0.8],
            [width * 0.7, height * 0.5],
            [width * 0.3, height * 0.5]
        ])
        dst = np.float32([
            [width * 0.1, height],
            [width * 0.9, height],
            [width * 0.9, 0],
            [width * 0.1, 0]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
        return warped, M

    def detect_yellow_line(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        return cv2.inRange(hsv, lower_yellow, upper_yellow)

    def detect_lanes(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        warped, M = self.perspective_transform(edges)
        histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)

        midpoint = len(histogram) // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        if histogram[leftx_base] < 100 or histogram[rightx_base] < 100:
            return None, None, None

        nwindows = 9
        window_height = warped.shape[0] // nwindows
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = warped.shape[0] - (window + 1) * window_height
            win_y_high = warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) == 0 or len(rightx) == 0:
            return None, None, None

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        lane_center = (left_fitx[-1] + right_fitx[-1]) // 2
        frame_center = warped.shape[1] // 2
        deviation = lane_center - frame_center

        return deviation, left_fitx, right_fitx

    def run(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to grab frame")
                break

            yellow_mask = self.detect_yellow_line(frame)
            yellow_pixels = np.sum(yellow_mask > 0)

            deviation, left_fitx, right_fitx = self.detect_lanes(frame)

            if deviation is not None:
                steering_angle = -deviation // 10
                steering_angle = np.clip(steering_angle, -30, 30)

                if yellow_pixels > 1000:
                    print("Warning: Approaching yellow line!")
                    steering_angle -= 10 if steering_angle > 0 else -10

                self.px.set_dir_servo_angle(steering_angle)
                self.px.forward(30)
                print(f"Steering Angle: {steering_angle}°")
            else:
                self.px.stop()
                print("No lanes detected. Stopping.")

            cv2.imshow('Lane Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()
        self.px.stop()

if __name__ == '__main__':
    detector = LaneDetection()
    detector.run()
