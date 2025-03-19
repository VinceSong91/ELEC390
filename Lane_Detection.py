import numpy as np
import cv2

# Region of interest selection
def region_selection(image):
    mask = np.zeros_like(image)
    ignore_mask_color = 255
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(image, mask)

# Hough transform for line detection
def hough_transform(image):
    # Ensure the input image is grayscale (CV_8UC1)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return cv2.HoughLinesP(image, 1, np.pi/180, 20, minLineLength=30, maxLineGap=500)

# Average slope and intercept for lane lines
def average_slope_intercept(lines):
    left_lines, right_lines = [], []
    left_weights, right_weights = [], []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:  # Vertical line, slope is undefined
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    # Compute weighted averages for left and right lanes
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

# Compute the pixel points for lane lines
def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    if slope == 0:  # Prevent divide by zero
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, int(y1)), (x2, int(y2))

# Draw the detected lane lines on the image
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

# Frame processor that includes edge detection, Hough transform, and lane drawing
def frame_processor(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    region = region_selection(edges)
    hough = hough_transform(region)

    # If no lines detected, return original image
    if hough is None or len(hough) == 0:
        return image

    # Get the average slope and intercept of detected lanes
    left_lane, right_lane = average_slope_intercept(hough)
    
    # Compute the pixel points for the lane lines
    left_line = pixel_points(image.shape[0], image.shape[0]*0.6, left_lane)
    right_line = pixel_points(image.shape[0], image.shape[0]*0.6, right_lane)

    # Draw the lane lines on the image
    result = draw_lane_lines(image, [left_line, right_line])
    return result

# Steering control to stay within the lanes
def steering_control(left_line, right_line, frame_width):
    # If no lanes are detected, stay centered
    if left_line is None or right_line is None:
        return "center"

    # Compute the center of the lane lines
    left_center = (left_line[0][0] + left_line[1][0]) / 2
    right_center = (right_line[0][0] + right_line[1][0]) / 2
    lane_center = (left_center + right_center) / 2
    
    # Compute the offset from the center of the frame
    offset = frame_width / 2 - lane_center
    
    # If offset is positive, steer right, if negative, steer left
    if offset > 0:
        return "right"
    elif offset < 0:
        return "left"
    else:
        return "center"

# Main function to run the video capture and lane detection
def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = frame_processor(frame)

        # Determine the lane's steering control
        left_lane, right_lane = average_slope_intercept(hough_transform(region_selection(frame)))
        control = steering_control(left_lane, right_lane, frame.shape[1])

        print(f"Steering control: {control}")

        cv2.imshow('Lane Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    os.environ['DISPLAY'] = ':0'
    main()
