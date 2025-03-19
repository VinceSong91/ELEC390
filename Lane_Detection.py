class MyHandCodedLaneFollower:
    def __init__(self, car):
        self.car = car
        self.curr_steering_angle = 90

    def follow_lane(self, image):
        # Your lane detection logic here (e.g., finding lanes and calculating steering angle)
        self.curr_steering_angle = self.calculate_steering_angle(image)
        print(f"Steering angle: {self.curr_steering_angle}")

        # Corrected the call to set the steering angle using Picarx
        self.car.px.set_dir_servo_angle(self.curr_steering_angle)

        return image

    def calculate_steering_angle(self, image):
        # Add your lane detection and steering angle logic here
        return 90
