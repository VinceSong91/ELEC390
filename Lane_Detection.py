import cv2
import numpy as np

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
    polygon = np.array([[
        (0, height),
        (width // 2 - 50, height // 2 + 50),
        (width // 2 + 50, height // 2 + 50),
        (width, height),
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
    
    return frame

# Capture video from the PiCar-X camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect lanes
    output_frame = detect_lanes(frame)
    
    # Display the output
    cv2.imshow("Lane Detection", output_frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()