import cv2
import numpy as np
import math
import copy

class LaneDetection:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)  # Use the default camera
        self.camera.set(3, 640)  # Set frame width
        self.camera.set(4, 480)  # Set frame height
        self.left_fit = None  # Store previous left lane fit
        self.right_fit = None  # Store previous right lane fit

    def preprocessing(self, img):
        """
        Preprocess the image to detect white and yellow lanes.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gblur = cv2.GaussianBlur(gray, (5, 5), 0)
        white_mask = cv2.threshold(gblur, 200, 255, cv2.THRESH_BINARY)[1]
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        return mask

    def region_of_interest(self, img, polygon):
        """
        Selects a region of interest (ROI) in the image.
        """
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [polygon], 255)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    def perspective_transform(self, img, src, dst, size):
        """
        Applies a bird's-eye view transformation to the image.
        """
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, size)
        return warped

    def sliding_window_search(self, binary_warped):
        """
        Detects lane pixels using a sliding window approach.
        """
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        margin = 100
        minpix = 50
        window_height = binary_warped.shape[0] // nwindows
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
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

        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=np.int64)
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=np.int64)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) == 0 or len(rightx) == 0:
            return None, None  # Return None if no lanes are detected

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def draw_lane_lines(self, img, left_fit, right_fit, src, dst, size):
        """
        Draws the detected lane lines on the image.
        """
        if left_fit is None or right_fit is None:
            return img  # Skip drawing if no lanes are detected

        ploty = np.linspace(0, size[1] - 1, size[1])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        warp_zero = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(warp_zero, np.int_([pts]), (0, 255, 0))
        M_inv = cv2.getPerspectiveTransform(dst, src)
        newwarp = cv2.warpPerspective(warp_zero, M_inv, (img.shape[1], img.shape[0]))
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        return result

    def run(self):
        """
        Main loop to capture video and process frames.
        """
        print("Press 'q' to stop.")
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture frame. Stopping.")
                    break

                # Preprocess the image
                processed_img = self.preprocessing(frame)

                # Define ROI polygon
                height, width = processed_img.shape
                polygon = np.array([[
                    [int(width * 0.1), height],
                    [int(width * 0.45), int(height * 0.6)],
                    [int(width * 0.55), int(height * 0.6)],
                    [int(0.95 * width), height]
                ]], dtype=np.int32)

                # Apply ROI masking
                masked_img = self.region_of_interest(processed_img, polygon)

                # Define source and destination points for perspective transform
                src = np.float32([[
                    [int(width * 0.45), int(height * 0.6)],
                    [int(width * 0.55), int(height * 0.6)],
                    [int(0.95 * width), height],
                    [int(width * 0.1), height]
                ]])
                dst = np.float32([[
                    [0, 0],
                    [width, 0],
                    [width, height],
                    [0, height]
                ]])
                warped_size = (width, height)

                # Apply perspective transform
                warped_img = self.perspective_transform(masked_img, src, dst, warped_size)

                # Detect lanes using sliding window search
                left_fit, right_fit = self.sliding_window_search(warped_img)

                # If no lanes are detected, use previous fit
                if left_fit is None or right_fit is None:
                    left_fit, right_fit = self.left_fit, self.right_fit
                else:
                    self.left_fit, self.right_fit = left_fit, right_fit  # Update previous fit

                # Draw lane lines on the original image
                result = self.draw_lane_lines(frame, left_fit, right_fit, src, dst, warped_size)

                # Display the result
                cv2.imshow("Lane Detection", result)

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopping.")
                    break
        except Exception as e:
            print(f"Error: {e}")
        finally:
            print("Cleaning up resources.")
            self.camera.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    LaneDetection().run()