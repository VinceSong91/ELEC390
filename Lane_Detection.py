import cv2
import numpy as np

class LaneDetection:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)  # Use the default camera
        self.camera.set(3, 640)  # Set frame width
        self.camera.set(4, 480)  # Set frame height

    def preprocessing(self, img):
        """
        Preprocess the image to detect white and yellow lanes.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gblur = cv2.GaussianBlur(gray, (5, 5), 0)
        white_mask = cv2.threshold(gblur, 200, 255, cv2.THRESH_BINARY)[1]
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        return mask

    def region_of_interest(self, img, polygon):
        """
        Selects a region of interest (ROI) in the image.
        """
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [polygon], 255)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    def run(self):
        """
        Main loop to capture video and process frames.
        """
        print("Press 'q' to stop.")
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture frame. Stopping.")
                    break

                # Display the original frame
                cv2.imshow("Original Frame", frame)

                # Preprocess the image
                processed_img = self.preprocessing(frame)

                # Define ROI polygon
                height, width = processed_img.shape
                polygon = np.array([[
                    [int(width * 0.1), height],  # Bottom-left
                    [int(width * 0.4), int(height * 0.6)],  # Top-left
                    [int(width * 0.6), int(height * 0.6)],  # Top-right
                    [int(0.9 * width), height]  # Bottom-right
                ]], dtype=np.int32)

                # Apply ROI masking
                masked_img = self.region_of_interest(processed_img, polygon)

                # Debug: Display intermediate results
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([30, 255, 255])
                yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
                combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

                cv2.imshow("Grayscale Image", gray)
                cv2.imshow("HSV Image", hsv)
                cv2.imshow("White Mask", white_mask)
                cv2.imshow("Yellow Mask", yellow_mask)
                cv2.imshow("Combined Mask", combined_mask)
                cv2.imshow("ROI Mask", masked_img)
                cv2.waitKey(1)

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopping.")
                    break
        except Exception as e:
            print(f"Error: {e}")
        finally:
            print("Cleaning up resources.")
            self.camera.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    LaneDetection().run()