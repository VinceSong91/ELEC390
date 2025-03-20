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

    # Clean up the masks using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)  # Remove noise
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)  # Fill gaps

    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)  # Remove noise
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)  # Fill gaps

    # Combine masks (optional, if you want a single mask for both lanes)
    mask_combined = cv2.bitwise_or(mask_white, mask_yellow)

    # Apply the masks to the transformed frame
    masked_white = cv2.bitwise_and(transformed_frame, transformed_frame, mask=mask_white)
    masked_yellow = cv2.bitwise_and(transformed_frame, transformed_frame, mask=mask_yellow)

    # Display frames for debugging
    cv2.imshow("Original", frame)
    cv2.imshow("Bird's Eye View", transformed_frame)
    cv2.imshow("HSV Transformed", hsv_transformed_frame)
    cv2.imshow("White Lane Mask", mask_white)
    cv2.imshow("Yellow Lane Mask", mask_yellow)
    cv2.imshow("Combined Mask", mask_combined)

    # Exit on 'ESC' key press
    if cv2.waitKey(10) == 27:
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()