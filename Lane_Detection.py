import cv2
import numpy as np

class LaneDetection:
    def __init__(self):
        self.camera = cv2.VideoCapture(-1)  # Use -1 for default camera
        self.camera.set(3, 320)
        self.camera.set(4, 240)

    def region_of_interest(self, image):
        mask = np.zeros_like(image)
        height, width = image.shape[:2]
        polygon = np.array([
            [(0, height), (width, height), (width // 2, height // 2)]
        ], np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        return cv2.bitwise_and(image, mask)

    def detect_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def detect_lines(self, edges, image):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image

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