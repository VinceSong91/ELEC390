import cv2
import numpy as np

class LaneDetection:
    def __init__(self):
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(3, 640)
        self.camera.set(4, 480)

    def region_of_interest(self, image):
        mask = np.zeros_like(image)
        height, width = image.shape[:2]
        polygons = np.array([
            [(0, height), (width, height), (width//2, height//2)]
        ])
        cv2.fillPoly(mask, polygons, (255, 255, 255))
        return cv2.bitwise_and(image, mask)

    def detect_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    def draw_lines(self, image, lines, color):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), color, 5)

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
