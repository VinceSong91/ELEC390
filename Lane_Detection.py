import numpy as np
import cv2

class LaneDetection:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)  # Change to 0 for default camera
        self.camera.set(3, 640)  # Set frame width
        self.camera.set(4, 480)  # Set frame height

    def region_selection(self, image):
        """
        Selects a region of interest (ROI) in the image.
        """
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
        """
        Applies Hough Transform to detect lines in the image.
        """
        return cv2.HoughLinesP(image, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=500)

    def filter_white_lanes(self, image):
        """
        Filters white lane lines using a high-intensity threshold.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return white_mask

    def filter_yellow_lanes(self, image):
        """
        Filters yellow lane lines using HSV color space.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow in HSV
        upper_yellow = np.array([30, 255, 255])  # Upper bound for yellow in HSV
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        return yellow_mask

    def detect_dotted_lines(self, lines):
        """
        Detects dotted lines by checking for small line segments with gaps.
        """
        dotted_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length < 50:  # Threshold for dotted line segments
                    dotted_lines.append(line)
        return dotted_lines

    def average_slope_intercept(self, lines):
        """
        Averages the slope and intercept of detected lines to form a single left and right lane.
        """
        left_lines = []
        right_lines = []
        if lines is not None:
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

    def draw_lane_lines(self, image, left_lane, right_lane, dotted_lines=None):
        """
        Draws the detected lane lines on the image.
        """
        line_image = np.zeros_like(image)
        for lane in [left_lane, right_lane]:
            if lane is not None:
                slope, intercept = lane
                y1 = image.shape[0]  # Bottom of the image
                y2 = int(y1 * 0.6)    # Top of the lane line

                # Avoid division by zero or near-zero slope
                if abs(slope) < 1e-5:
                    slope = 1e-5  # Set a small value to avoid division by zero
                
                x1 = int((y1 - intercept) / slope)
                x2 = int((y2 - intercept) / slope)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  # Draw solid lines

        # Draw dotted lines
        if dotted_lines is not None:
            for line in dotted_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 5)  # Draw dotted lines in yellow

        return cv2.addWeighted(image, 0.8, line_image, 1, 0)

    def follow_lane(self, image):
        """
        Processes the image to detect and follow the lane.
        """
        # Detect white lanes
        white_mask = self.filter_white_lanes(image)
        white_edges = cv2.Canny(white_mask, 50, 150)
        white_region = self.region_selection(white_edges)
        white_lines = self.hough_transform(white_region)

        # Detect yellow lanes
        yellow_mask = self.filter_yellow_lanes(image)
        yellow_edges = cv2.Canny(yellow_mask, 50, 150)
        yellow_region = self.region_selection(yellow_edges)
        yellow_lines = self.hough_transform(yellow_region)

        # Separate solid and dotted lines
        dotted_lines = self.detect_dotted_lines(yellow_lines)

        # Average solid lines
        left_lane, right_lane = self.average_slope_intercept(white_lines)

        # Draw lanes
        return self.draw_lane_lines(image, left_lane, right_lane, dotted_lines)

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
                
                processed_frame = self.follow_lane(frame)

                # Resize the frame to make it larger
                scale_percent = 150  # Increase this value to make the window larger
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                resized_frame = cv2.resize(processed_frame, (width, height), interpolation=cv2.INTER_AREA)

                cv2.imshow("Lane Detection", resized_frame)

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