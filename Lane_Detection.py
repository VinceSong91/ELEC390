import numpy as np
import cv2

class LaneDetection:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)  # Change to 0 for default camera
        self.camera.set(3, 640)  # Set frame width
        self.camera.set(4, 480)  # Set frame height
        self.left_fit = None  # Store previous left lane fit
        self.right_fit = None  # Store previous right lane fit

    def region_selection(self, image):
        """
        Selects a region of interest (ROI) in the image.
        """
        mask = np.zeros_like(image)
        rows, cols = image.shape[:2]
        # Define a trapezoidal ROI (adjust these values based on your camera's perspective)
        vertices = np.array([[
            [cols * 0.1, rows * 0.95],  # Bottom-left
            [cols * 0.4, rows * 0.6],   # Top-left
            [cols * 0.6, rows * 0.6],   # Top-right
            [cols * 0.9, rows * 0.95]   # Bottom-right
        ]], dtype=np.int32)
        cv2.fillPoly(mask, [vertices], 255)
        return cv2.bitwise_and(image, mask)

    def color_filter(self, image):
        """
        Filters white and yellow lane markings using color thresholds.
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for white and yellow
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Create masks for white and yellow
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        return cv2.bitwise_and(image, image, mask=combined_mask)

    def perspective_transform(self, image):
        """
        Applies a bird's-eye view transformation to the image.
        """
        rows, cols = image.shape[:2]
        # Define source and destination points for the transform
        src = np.float32([[
            [cols * 0.1, rows * 0.95],  # Bottom-left
            [cols * 0.4, rows * 0.6],   # Top-left
            [cols * 0.6, rows * 0.6],   # Top-right
            [cols * 0.9, rows * 0.95]   # Bottom-right
        ]])
        dst = np.float32([[
            [cols * 0.2, rows],  # Bottom-left
            [cols * 0.2, 0],     # Top-left
            [cols * 0.8, 0],     # Top-right
            [cols * 0.8, rows]   # Bottom-right
        ]])
        # Compute the perspective transform matrix
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)  # Inverse transform
        # Warp the image
        warped = cv2.warpPerspective(image, self.M, (cols, rows))
        return warped

    def sliding_window_search(self, binary_warped):
        """
        Detects lane pixels using a sliding window approach.
        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # Find the peak of the left and right halves of the histogram
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set up sliding windows
        nwindows = 9
        window_height = binary_warped.shape[0] // nwindows
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            # Identify window boundaries
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If found > minpix pixels, recenter next window
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=np.int64)
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=np.int64)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Check if lanes are detected
        if len(leftx) == 0 or len(rightx) == 0:
            return None, None  # Return None if no lanes are detected

        # Fit a second-order polynomial to each lane
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def draw_lane_lines(self, image, left_fit, right_fit):
        """
        Draws the detected lane lines on the image.
        """
        if left_fit is None or right_fit is None:
            return image  # Skip drawing if no lanes are detected

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(image[:, :, 0]).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space
        newwarp = cv2.warpPerspective(color_warp, self.M_inv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        return cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    def follow_lane(self, image):
        """
        Processes the image to detect and follow the lane.
        """
        # Apply color filtering
        filtered = self.color_filter(image)

        # Apply perspective transform
        warped = self.perspective_transform(filtered)

        # Convert to binary image
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, binary_warped = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Detect lanes using sliding window search
        left_fit, right_fit = self.sliding_window_search(binary_warped)

        # If no lanes are detected, use previous fit
        if left_fit is None or right_fit is None:
            left_fit, right_fit = self.left_fit, self.right_fit
        else:
            self.left_fit, self.right_fit = left_fit, right_fit  # Update previous fit

        # Draw lane lines on the original image
        return self.draw_lane_lines(image, left_fit, right_fit)

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
                
                # Process the frame to detect lanes
                processed_frame = self.follow_lane(frame)

                # Display the processed frame
                cv2.imshow("Lane Detection", processed_frame)

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