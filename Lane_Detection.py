import cv2
import numpy as np
from picarx import Picarx

# Initialize PiCar-X
px = Picarx()

# Initialize PiCar-X camera
camera = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Capture frame from PiCar-X camera
    success, frame = camera.read()
    if not success:
        print("Failed to capture frame from camera.")
        break

    frame = cv2.resize(frame, (640, 480))

    ## Choosing points for perspective transformation
    tl = (222, 387)  # Top-left
    bl = (70, 472)   # Bottom-left
    tr = (400, 380)  # Top-right
    br = (538, 472)  # Bottom-right

    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)

    ## Applying perspective transformation
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    ### Object Detection
    # Convert the transformed frame to HSV color space
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for white and yellow
    # White lanes
    lower_white = np.array([0, 0, 200])  # Low H, S, high V
    upper_white = np.array([179, 30, 255])  # High H, low S, high V

    # Yellow lanes
    lower_yellow = np.array([20, 100, 100])  # Low H, high S, high V
    upper_yellow = np.array([30, 255, 255])  # High H, high S, high V

    # Create masks for white and yellow
    mask_white = cv2.inRange(hsv_transformed_frame, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv_transformed_frame, lower_yellow, upper_yellow)

    # Combine masks
    mask = cv2.bitwise_or(mask_white, mask_yellow)

    # Clean up the mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill gaps

    # Histogram to find lane positions
    histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding Window
    y = 472
    lx = []
    rx = []

    msk = mask.copy()

    while y > 0:
        ## Left threshold
        img = mask[y - 40:y, left_base - 50:left_base + 50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lx.append(left_base - 50 + cx)
                left_base = left_base - 50 + cx

        ## Right threshold
        img = mask[y - 40:y, right_base - 50:right_base + 50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lx.append(right_base - 50 + cx)
                right_base = right_base - 50 + cx

        cv2.rectangle(msk, (left_base - 50, y), (left_base + 50, y - 40), (255, 255, 255), 2)
        cv2.rectangle(msk, (right_base - 50, y), (right_base + 50, y - 40), (255, 255, 255), 2)
        y -= 40

    # Display frames
    cv2.imshow("Original", frame)
    cv2.imshow("Bird's Eye View", transformed_frame)
    cv2.imshow("Lane Detection - Mask", mask)
    cv2.imshow("Lane Detection - Sliding Windows", msk)

    # Exit on 'ESC' key press
    if cv2.waitKey(10) == 27:
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()