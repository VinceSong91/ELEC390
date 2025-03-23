import cv2
import numpy as np
import picar

class LaneDetection:
    def __init__(self):
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(3, 640)
        self.camera.set(4, 480)
        picar.setup()
        self.servo = picar.Servo()

    def region_of_interest(self, image):
        mask = np.zeros_like(image)
        height, width = image.shape[:2]
        polygons = np.array([
            [(0, height), (width, height), (int(width * 0.45), int(height * 0.6)), (int(width * 0.55), int(height * 0.6))]
        ])
        cv2.fillPoly(mask, polygons, (255, 255, 255))
        return cv2.bitwise_and(image, mask)

    def detect_edges(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Improved masking for white and yellow
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))

        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        filtered_image = cv2.bitwise_and(image, image, mask=combined_mask)

        gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def detect_lines(self, edges, image):
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=150)
        left_lines = []
        right_lines = []
        middle_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)

                if abs(slope) < 0.3: # Middle yellow line (dotted)
                    middle_lines.append(line)
                elif slope < 0: # Left white line
                    left_lines.append(line)
                else: # Right white line
                    right_lines.append(line)
        
        self.draw_lines(image, left_lines, (255, 255, 255))  # White for solid outer lines
        self.draw_lines(image, right_lines, (255, 255, 255))
        self.draw_lines(image, middle_lines, (0, 255, 255)) # Yellow for the middle line
        
        self.plot_center_point(image, left_lines, right_lines)

    def draw_lines(self, image, lines, color):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), color, 5)

    def plot_center_point(self, image, left_lines, right_lines):
        left_xs = [line[0][0] for line in left_lines if len(line[0]) == 4]
        right_xs = [line[0][0] for line in right_lines if len(line[0]) == 4]

        if left_xs and right_xs:
            left_edge = max(left_xs)
            right_edge = min(right_xs)
            center_x = (left_edge + right_edge) // 2
            height = image.shape[0]
            center_y = int(height * 0.7)

            cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), -1)

            # Control car wheels to align with midpoint using picar
            frame_center = image.shape[1] // 2
            steering_angle = int((center_x - frame_center) / 3)
            steering_angle = max(-45, min(45, steering_angle))

            print(f"Steering Angle: {steering_angle}")
            self.servo.set_angle(steering_angle)

    def run(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break

            roi = self.region_of_interest(frame)
            edges = self.detect_edges(roi)
            self.detect_lines(edges, frame)

            cv2.imshow('Lane Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    lane_detector = LaneDetection()
    lane_detector.run()
