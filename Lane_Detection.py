import cv2
import numpy as np
import time
from picarx import Picarx  # Assuming you have the Picar-X library installed

# Initialize PiCar-X
px = Picarx()

# Ultrasonic sensor setup
def get_distance():
    return px.ultrasonic.read()

# Function to detect lanes
def detect_lanes(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny
    edges = cv2.Canny(blur, 50, 150)
    
    # Define region of interest (ROI)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    
    # Adjusted polygon for downward and rightward tilt
    polygon = np.array([[
        (width * 0.4, height * 0.6),  # Top-left point shifted right
        (width * 0.1, height),        # Bottom-left point
        (width, height),              # Bottom-right point
        (width * 0.7, height * 0.6),  # Top-right point shifted left
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=100)
    
    # Separate and classify lines
    left_lines = []
    right_lines = []
    middle_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            
            # Classify lines based on slope and position
            if slope < -0.5:  # Left lane line
                left_lines.append(line[0])
            elif slope > 0.5:  # Right lane line
                right_lines.append(line[0])
            else:  # Middle dotted line
                middle_lines.append(line[0])
    
    # Draw detected lines on the frame
    def draw_lines(img, lines, color, thickness):
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    draw_lines(frame, left_lines, (255, 0, 0), 5)  # Blue for left lane
    draw_lines(frame, right_lines, (0, 0, 255), 5)  # Red for right lane
    draw_lines(frame, middle_lines, (0, 255, 255), 5)  # Yellow for middle lane
    
    return frame, left_lines, right_lines, middle_lines

# Function to adjust camera tilt and wheel steering
def adjust_camera_and_wheels(left_lines, right_lines, distance):
    if distance < 20:  # Obstacle too close, stop the car
        px.stop()
        print("Obstacle detected! Stopping the car.")
        return
    
    # Calculate the center of the detected lanes
    left_lane_pos = np.mean([line[0] for line in left_lines]) if left_lines else None
    right_lane_pos = np.mean([line[0] for line in right_lines]) if right_lines else None
    
    if left_lane_pos is not None and right_lane_pos is not None:
        lane_center = (left_lane_pos + right_lane_pos) / 2
        frame_center = 320  # Assuming frame width is 640px
        
        # Calculate deviation from the center
        deviation = lane_center - frame_center
        
        # Adjust wheel steering based on deviation
        steering_angle = -deviation / 100  # Scale deviation to a reasonable steering angle
        px.set_dir_servo_angle(steering_angle)
        print(f"Steering angle: {steering_angle:.2f}")
        
        # Adjust camera tilt based on deviation
        camera_tilt = -deviation / 200  # Scale deviation to a reasonable tilt angle
        px.set_cam_pan_angle(camera_tilt)
        print(f"Camera tilt: {camera_tilt:.2f}")
    else:
        # No lanes detected, stop the car
        px.stop()
        print("No lanes detected! Stopping the car.")

# Capture video from the PiCar-X camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect lanes
    output_frame, left_lines, right_lines, middle_lines = detect_lanes(frame)
    
    # Get distance from ultrasonic sensor
    distance = get_distance()
    print(f"Distance: {distance} cm")
    
    # Adjust camera tilt and wheel steering
    adjust_camera_and_wheels(left_lines, right_lines, distance)
    
    # Display the output
    cv2.imshow("Lane Detection", output_frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
px.stop()