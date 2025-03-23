import cv2
import numpy as np
from picarx import Picarx
import time

class LaneDetection:
    def __init__(self):
        self.camera = cv2.VideoCapture(-1)  # Use -1 for default camera
        self.camera.set(3, 640)  # Set width to 640
        self.camera.set(4, 480)  # Set height to 480

    def perspective_transform(self, image):
        height, width = image.shape[:2]
        # Define source and destination points for the transform
        src = np.float32([
            [width * 0.4, height * 0.6],
            [width * 0.6, height * 0.6],
            [width * 0.9, height],
            [width * 0.1, height]
        ])
        dst = np.float32([
            [width * 0.1, 0],
            [width * 0.9, 0],
            [width * 0.9, height],
            [width * 0.1, height]
        ])
        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image
        warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
        return warped, M

    def detect_lanes(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Detect edges using Canny
        edges = cv2.Canny(blur, 50, 150)
        # Apply perspective transform
        warped, _ = self.perspective_transform(edges)
        # Use sliding windows to detect lanes
        histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set up sliding windows
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
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second-order polynomial to the lane lines
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        # Calculate the center of the lane
        lane_center = (left_fitx[-1] + right_fitx[-1]) // 2
        frame_center = warped.shape[1] // 2
        deviation = lane_center - frame_center

        return deviation

    def run(self, px):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process the frame to detect lanes
            deviation = self.detect_lanes(frame)

            if deviation is not None:
                # Adjust steering angle based on deviation
                steering_angle = -deviation // 10  # Scale deviation to steering angle
                steering_angle = max(-30, min(30, steering_angle))  # Constrain steering angle
                px.set_dir_servo_angle(steering_angle)

                # Move forward
                px.forward(30)
            else:
                # Stop if no lanes are detected
                px.stop()

            # Display the frame
            cv2.imshow('Lane Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()
        px.stop()

if __name__ == '__main__':
    px = Picarx()
    lane_detector = LaneDetection()
    lane_detector.run(px)