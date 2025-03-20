import numpy as np
import cv2

class LaneDetection:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)  # Change to 0 for default camera
        self.camera.set(3, 320)  # Set frame width
        self.camera.set(4, 240)  # Set frame height

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

    def draw_lane_lines(self, image, left_lane, right_lane):
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
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return cv2.addWeighted(image, 0.8, line_image, 1, 0)

    def follow_lane(self, image):
        """
        Processes the image to detect and follow the lane.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        region = self.region_selection(edges)
        lines = self.hough_transform(region)
        left_lane, right_lane = self.average_slope_intercept(lines)
        return self.draw_lane_lines(image, left_lane, right_lane)

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
                scale_percent = 200  # Increase this value to make the window larger
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
        finally