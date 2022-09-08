import numpy as np
import time

class PIDControl:

    def __init__(self):
        """Initialises class variables
        """
        self.radians_per_elem = None

        # angle between the two lidar ranges we are measuring
        self.theta = np.pi / 3 # 60 deg

        # angle of the car relative to the centre line
        self.alpha = 0

        # distance we want the car to be from the wall
        self.wall_dist = 1.75

        # a correction factor for the time, since we initialise the class before the actual simulation starts
        self.prev_time = time.time() + 1.7
        self.prev_error = 0
        self.total_error = 0

    def find_ranges(self, ranges):
        """Finds two LiDAR distances, a and b
        b is the LiDAR distance perpendicular to the car's axis
        a is the LiDAR distance which is at an angle theta relative to b"""
        
        # we want the distance directly to the right of the car (relative to the front of the car)
        quarter = int(len(ranges) / 4)
        b = ranges[quarter]

        index_a = quarter + int(self.theta / self.radians_per_elem)
        a = ranges[index_a]

       # print ("a = {}, b = {}".format(a, b))

        return a, b

    def find_alpha(self, a, b):
        """ Finds the angle alpha, which is the angle at which the car is to the centre of the track
        """
        p = (a * np.cos(self.theta)) - b
        q = a * np.sin(self.theta)

        self.alpha = np.arctan(p / q)

        return

    def find_future_distance(self, ranges):
        """ Finds the predicted distance that the car will be from the wall after
        travelling a distance L
        """

        # the distance we want to "look ahead"
        L = 1

        a, b = self.find_ranges(ranges)

        self.find_alpha(a, b)

        # current distance from the wall
        Dt = b * np.cos(self.alpha)

        future_dist = Dt + (L * np.sin(self.alpha))

        return future_dist

    def PID(self, ranges):
        """ Calculates the steering angle using a Proportional Integral Derivative (PID) controller
        """
        # PID constants
        Kp = 0.3
        Ki = 0 #0.00012
        Kd = 0.01

        time_elapsed = time.time() - self.prev_time
        self.prev_time = time.time()

        actual_wall_dist = self.find_future_distance(ranges)

        error = self.wall_dist - actual_wall_dist

        # our "integral"
        self.total_error += error

        # our "derivative"
        derror_dt = (error - self.prev_error) / time_elapsed

        steering_angle = (Kp * error) + (Kd* derror_dt) + (Ki * self.total_error)
        self.prev_error = error

        return steering_angle

    def calc_speed(self):
        """Calculate the speed at which the car should travel based on the angle 
        at which it is to the centre of the track
        """
        if 0 <= self.alpha and self.alpha < 0.1745329: # between 0 and 10 deg
            speed = 7
        elif  0.1745329 <= self.alpha and self.alpha < 0.3490659: # between 10 and 20 deg
            speed = 5
        else:
            speed = 3

        return speed

    def process_lidar(self, ranges):
        """ Process each LiDAR scan as per the PID Wall Follower algorithm & publish an AckermannDriveStamped Message
        """
        self.radians_per_elem = (2*np.pi) / len(ranges)
        steering_angle = self.PID(ranges)
        speed = self.calc_speed()

        return speed, steering_angle