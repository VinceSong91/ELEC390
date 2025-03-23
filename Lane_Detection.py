import cv2
import numpy as np
from picarx import Picarx  # Import the PiCar-X library

# Initialize PiCar-X
px = Picarx()

# Function to detect lanes
def detect_lanes(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny
    edges = cv2.Canny(blur, 50, 150)
    
    # Define region of interest (ROI) for a downward-tilted camera
    height, width = edges.shape
    mask = np.zeros_like(edges)
    
    # Adjusted polygon for a more downward-tilted camera
    polygon = np.array([[
        (width * 0.3, height * 0.5),  # Top-left point shifted down and right
        (width * 0.1, height),        # Bottom-left point
        (width * 0.9, height),        # Bottom-right point
        (width * 0.7, height * 0.5),  # Top-right point shifted down and left
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=100)
    
    # Separate and classify lines
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            
            # Classify lines based on slope and position (only right lane)
            if slope > 0.5:  # Right lane line
                right_lines.append(line[0])
    
    # Draw detected lines on the frame
    def draw_lines(img, lines, color, thickness):
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    draw_lines(frame, right_lines, (0, 0, 255), 5)  # Red for right lane
    
    return frame, right_lines

# Function to adjust wheel steering and camera tilt based on the right lane
def follow_right_lane(right_lines):
    if right_lines:
        # Calculate the average x-position of the right lane
        right_lane_pos = np.mean([line[0] for line in right_lines])
        frame_center = 320  # Assuming frame width is 640px
        
        # Calculate deviation from the center
        deviation = right_lane_pos - frame_center
        
        # Adjust wheel steering based on deviation
        steering_angle = -deviation / 50 - 15  # Increase sensitivity for steering
        px.set_dir_servo_angle(steering_angle)
        print(f"Steering angle: {steering_angle:.2f}")
        
        # Adjust camera tilt to keep the right lane centered
        camera_tilt = -deviation / 100  # Increase sensitivity for camera tilt
        px.set_cam_tilt_angle(camera_tilt)
        print(f"Camera tilt: {camera_tilt:.2f}")
        
        # Move forward
        px.forward(50)
    else:
        # No right lane detected, stop the car
        px.stop()
        print("No right lane detected! Stopping the car.")

# Capture video from the PiCar-X camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect lanes
    output_frame, right_lines = detect_lanes(frame)
    
    # Follow the right lane
    follow_right_lane(right_lines)
    
    # Display the output
    cv2.imshow("Lane Detection", output_frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
px.stop()