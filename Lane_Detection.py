import cv2
import numpy as np

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

        # Adjusted HSV range for yellow
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Use adaptive thresholding for white lanes
        white_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Combine white and yellow masks
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

    def sliding_window_search(self, binary_warped):
        """
        Detects lane pixels using a sliding window approach.
        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])  # Starting x-position for the left lane
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint  # Starting x-position for the right lane

        # Create an output image to visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # Set up sliding windows
        nwindows = 9  # Number of sliding windows
        margin = 100  # Width of the windows
        minpix = 50  # Minimum number of pixels to recenter the window
        window_height = binary_warped.shape[0] // nwindows  # Height of each window

        # Identify all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])  # y-coordinates of nonzero pixels
        nonzerox = np.array(nonzero[1])  # x-coordinates of nonzero pixels

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Lists to store lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            # Identify window boundaries
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height  # Bottom of the window
            win_y_high = binary_warped.shape[0] - window * window_height  # Top of the window
            win_xleft_low = leftx_current - margin  # Left boundary of the left window
            win_xleft_high = leftx_current + margin  # Right boundary of the left window
            win_xright_low = rightx_current - margin  # Left boundary of the right window
            win_xright_high = rightx_current + margin  # Right boundary of the right window

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify nonzero pixels within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If enough pixels are found, recenter the window
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=np.int64)
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=np.int64)

        # Extract left and right lane pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Visualize the sliding windows and detected lane pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]  # Red for left lane
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]  # Blue for right lane
        cv2.imshow("Sliding Windows", out_img)

        if len(leftx) == 0 or len(rightx) == 0:
            return None, None  # Return None if no lanes are detected

        # Fit a second-order polynomial to the lane pixels
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def draw_lane_lines(self, img, left_fit, right_fit):
        """
        Draws the detected lane lines and fills the space between them.
        """
        if left_fit is None or right_fit is None:
            return img  # Skip drawing if no lanes are detected

        # Generate points for the left and right lanes
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        # Create an image to draw the lines on
        line_image = np.zeros_like(img)

        # Draw the lane lines
        for x1, y1, x2, y2 in zip(left_fitx, ploty, right_fitx, ploty):
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)

        # Fill the space between the lanes
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(line_image, np.int_([pts]), (0, 255, 0))

        # Overlay the lane lines and filled area on the original image
        result = cv2.addWeighted(img, 1, line_image, 0.3, 0)
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
                    [int(width * 0.1), height],  # Bottom-left
                    [int(width * 0.4), int(height * 0.6)],  # Top-left
                    [int(width * 0.6), int(height * 0.6)],  # Top-right
                    [int(0.9 * width), height]  # Bottom-right
                ]], dtype=np.int32)

                # Apply ROI masking
                masked_img = self.region_of_interest(processed_img, polygon)

                # Detect lanes using sliding window search
                left_fit, right_fit = self.sliding_window_search(masked_img)

                # If no lanes are detected, use previous fit
                if left_fit is None or right_fit is None:
                    left_fit, right_fit = self.left_fit, self.right_fit
                else:
                    self.left_fit, self.right_fit = left_fit, right_fit  # Update previous fit

                # Draw lane lines and fill the space between them
                result = self.draw_lane_lines(frame, left_fit, right_fit)

                # Display the result
                cv2.imshow("Lane Detection", result)

                # Debug: Display intermediate results
                cv2.imshow("Preprocessed Image", processed_img)
                cv2.imshow("ROI Mask", masked_img)
                cv2.waitKey(1)

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