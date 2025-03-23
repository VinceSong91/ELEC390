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
            [width * 0.2, height * 0.8],  # Bottom-left
            [width * 0.8, height * 0.8],  # Bottom-right
            [width * 0.7, height * 0.5],  # Top-right
            [width * 0.3, height * 0.5]   # Top-left
        ])
        dst = np.float32([
            [width * 0.1, height],  # Bottom-left
            [width * 0.9, height],  # Bottom-right
            [width * 0.9, 0],      # Top-right
            [width * 0.1, 0]       # Top-left
        ])
        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image
        warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
        return warped, M

    def detect_yellow_line(self, image):
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Define range for yellow color in HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        # Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        return mask

    def detect_lanes(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Detect edges using Canny
        edges = cv2.Canny(blur, 50, 150)
        # Apply perspective transform
        warped, M = self.perspective_transform(edges)
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

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Check if any lane pixels were detected
        if len(leftx) == 0 or len(rightx) == 0:
            return None, None, None  # No lanes detected

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

        return deviation, left_fitx, right_fitx

    def draw_lanes(self, frame, left_fitx, right_fitx, steering_angle):
        height, width, _ = frame.shape
        ploty = np.linspace(0, height - 1, height)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        _, Minv = self.perspective_transform(frame)

        # Draw the left and right lane lines
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane lines on the frame
        cv2.polylines(frame, [np.int32(pts_left)], isClosed=False, color=(0, 255, 0), thickness=5)
        cv2.polylines(frame, [np.int32(pts_right)], isClosed=False, color=(0, 255, 0), thickness=5)

        # Draw the steering direction
        center_x = width // 2
        center_y = height
        end_x = center_x + int(100 * np.sin(np.radians(steering_angle)))
        end_y = center_y - int(100 * np.cos(np.radians(steering_angle)))
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 5)

        return frame

    def run(self, px):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Detect yellow dotted line
            yellow_mask = self.detect_yellow_line(frame)
            yellow_pixels = np.sum(yellow_mask > 0)

            # Process the frame to detect lanes
            deviation, left_fitx, right_fitx = self.detect_lanes(frame)

            if deviation is not None:
                # Adjust steering angle based on deviation
                steering_angle = -deviation // 10  # Scale deviation to steering angle
                steering_angle = max(-30, min(30, steering_angle))  # Constrain steering angle

                # If yellow line is detected and car is too close, steer away
                if yellow_pixels > 1000:  # Threshold for yellow line detection
                    # Calculate the distance to the yellow line
                    yellow_line_pos = np.mean(np.where(yellow_mask > 0)[1])
                    car_pos = frame.shape[1] // 2
                    distance_to_yellow = car_pos - yellow_line_pos

                    # If car is too close to the yellow line, steer away
                    if distance_to_yellow < 50:  # Safety margin
                        steering_angle = -20  # Steer right to avoid crossing the yellow line
                        print("Warning: Too close to yellow line! Steering right.")

                px.set_dir_servo_angle(steering_angle)

                # Pan the camera to follow the steering direction
                px.set_cam_pan_angle(steering_angle // 2)  # Adjust the camera pan angle

                # Move forward
                px.forward(30)

                # Draw the detected lanes and steering direction
                frame = self.draw_lanes(frame, left_fitx, right_fitx, steering_angle)

                # Terminal updates
                print(f"Steering Angle: {steering_angle}°")
                print(f"Yellow Line Detected: {yellow_pixels > 1000}")
            else:
                # Stop if no lanes are detected
                px.stop()
                print("No lanes detected. Stopping.")

            # Display the frame
            cv2.imshow('Lane Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()
        px.stop()

if __name__ == '__main__':
    # Initialize Picar-X with corrected servo and motor offsets
    px = Picarx()
    px.set_dir_servo_offset(0)  # Reset direction servo offset to 0
    px.set_motor_offset(1, 1)   # Ensure motors are calibrated correctly

    lane_detector = LaneDetection()
    lane_detector.run(px)