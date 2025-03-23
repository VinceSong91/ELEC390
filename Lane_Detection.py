import cv2
import numpy as np

class LaneDetection:
    def __init__(self):
        self.camera = cv2.VideoCapture(-1)  # Use -1 for default camera
        self.camera.set(3, 640)  # Set width to 640
        self.camera.set(4, 480)  # Set height to 480

    def region_of_interest(self, image):
        height, width = image.shape[:2]
        mask = np.zeros_like(image)
        polygon = np.array([
            [(0, height), (width, height), (width // 2 + 50, height // 2 + 50), (width // 2 - 50, height // 2 + 50)]
        ], np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        return cv2.bitwise_and(image, mask)

    def detect_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def average_slope_intercept(self, image, lines):
        left_fit = []
        right_fit = []
        if lines is None:
            return None

        for line in lines:
            x1, y1, x2, y2 = line[0]
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        left_fit_avg = np.average(left_fit, axis=0) if left_fit else None
        right_fit_avg = np.average(right_fit, axis=0) if right_fit else None

        return left_fit_avg, right_fit_avg

    def make_line_points(self, image, line_parameters):
        if line_parameters is None:
            return None

        slope, intercept = line_parameters
        height, width, _ = image.shape

        # Calculate y1 (bottom of the image) and y2 (slightly above the middle)
        y1 = height
        y2 = int(height * 0.6)

        # Calculate x1 and x2 using the line equation: x = (y - intercept) / slope
        if slope == 0:  # Avoid division by zero
            return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        # Ensure the points are within the image boundaries
        if x1 < 0 or x1 > width or x2 < 0 or x2 > width:
            return None

        return ((x1, y1), (x2, y2))  # Return a tuple of tuples

    def detect_lines(self, edges, image):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
        averaged_lines = self.average_slope_intercept(image, lines)
        line_image = np.zeros_like(image)

        if averaged_lines is not None:
            left_line = self.make_line_points(image, averaged_lines[0])
            right_line = self.make_line_points(image, averaged_lines[1])

            if left_line is not None:
                print("Left Line Points:", left_line)  # Debugging
                cv2.line(line_image, left_line[0], left_line[1], (0, 255, 0), 10)
            if right_line is not None:
                print("Right Line Points:", right_line)  # Debugging
                cv2.line(line_image, right_line[0], right_line[1], (0, 255, 0), 10)

            # Calculate the center of the lane
            if left_line is not None and right_line is not None:
                center_x = (left_line[0][0] + right_line[0][0]) // 2
                center_y = (left_line[0][1] + right_line[0][1]) // 2
                cv2.circle(line_image, (center_x, center_y), 10, (0, 0, 255), -1)

        return cv2.addWeighted(image, 0.8, line_image, 1, 1)

    def process_frame(self, frame):
        edges = self.detect_edges(frame)
        masked_edges = self.region_of_interest(edges)
        return self.detect_lines(masked_edges, frame)

    def run(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to grab frame")
                break
            processed_frame = self.process_frame(frame)
            cv2.imshow('Lane Detection', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    lane_detector = LaneDetection()
    lane_detector.run()