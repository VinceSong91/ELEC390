import cv2
import numpy as np

def region_of_interest(image):
    height, width = image.shape[:2]
    mask = np.zeros_like(image)

    # Define region for lane detection (trapezoid shape)
    polygons = np.array([
        [(0, height), (width, height), (width//2, height//2)]
    ])

    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return line_image

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150
    )

    line_image = draw_lines(frame, lines)
    combined = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    return combined

def main():
    cap = cv2.VideoCapture(-1)  # Use default camera
    cap.set(3, 320)  # Width
    cap.set(4, 240)  # Height

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        processed_frame = process_frame(frame)

        cv2.imshow("Lane Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()