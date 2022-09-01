import numpy as np
import time

class PIDControl:

    SPEED = 3.0

    def __init__(self):
        """Initialises class variables
        """
        self.radians_per_elem = None
        self.centre_elem = None
        self.prev_time = time.time() + 1.7
        self.prev_error = 0
        self.total_error = 0
        self.side = None


    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array by removing the LiDAR scan data from
        directly in front of and behind the car. Returns two arrays: the LiDAR scans to the left 
        and to the right of the car
        """
        self.radians_per_elem = (2*np.pi) / len(ranges)
        eighth = int(len(ranges) / 8)
        # LiDAR scan data in the right quadrant
        right_ranges = np.array(ranges[eighth: (3*eighth)])
        # LiDAR scan data in the left quadrant
        left_ranges = np.array(ranges[-(3 * eighth): -eighth])
        return left_ranges, right_ranges


    def find_max_gap(self, l_ranges, r_ranges):
        """Finds the width of the racetrack given the left and right distances.
        Returns the gap width and the distance to the closest wall
        """
        min_l_dist, min_l_elem = self.find_min_distance(l_ranges)
        min_r_dist, min_r_elem = self.find_min_distance(r_ranges)
        if (min_l_dist < min_r_dist):
            gap_width = min_l_dist + r_ranges[min_l_elem]
            self.side = "L"
            return gap_width, min_l_dist
            #print("Left = {}, right = {}".format(min_l_dist, r_ranges[min_r_elem]))
        else:
            gap_width = min_r_dist + l_ranges[min_r_elem]
            self.side = "R"
            return gap_width, min_r_dist
            #print("Left = {}, right = {}".format(l_ranges[min_r_elem], min_l_dist))
        

    def find_min_distance(self, ranges):
        """Finds the closest LiDAR distance and returns both the distance and the index of that
        distance in the array of LiDAR distances
        """
        min_dist = ranges[0]
        min_elem_num = 0
        for i, dist in enumerate(ranges):
            if dist > 0:
                if dist < min_dist:
                    min_dist = dist
                    min_elem_num = i

        return min_dist, min_elem_num


    def find_angle(self, ranges):
        """ Find the angle of the car relative to the centre line of the track
        """
        if self.side == "L":
            min_dist, min_elem = self.find_min_distance(ranges)
            centre = int(len(ranges) / 2)
            return (centre - min_elem) * self.radians_per_elem
        else:
            min_dist, min_elem = self.find_min_distance(ranges)
            centre = int(len(ranges) / 2)
            return (min_elem - centre) * self.radians_per_elem
        

    def PID(self, l_ranges, r_ranges):
        """ Return the steering angle given the LiDAR distances to the left and right of the car
        """

        # PID constants
        Kp = 0.2 
        Ki = 0.00012
        Kd = 0.0002

        gap_width, min_dist = self.find_max_gap(l_ranges, r_ranges)

        gap_centre = gap_width / 2
        time_elapsed = time.time() - self.prev_time
        self.prev_time = time.time()

        if self.side == "L": 
            y = min_dist - gap_centre
            theta = self.find_angle(l_ranges)   
        elif self.side == "R":
            y = gap_centre - min_dist
            theta = self.find_angle(r_ranges)  


        error = y + ((self.SPEED * time_elapsed) * np.sin(theta))
        self.total_error += error
        derror_dt = (error - self.prev_error) / time_elapsed

        steering_angle = (Kp * error) + (Kd* derror_dt) + (Ki * self.total_error)
        self.prev_error = error

        return steering_angle

    
    def process_lidar(self, ranges):
        """ Process each LiDAR scan as per the PID Wall Follow algorithm & publish an AckermannDriveStamped Message
        """
        left_ranges, right_ranges = self.preprocess_lidar(ranges)
        
        steering_angle = self.PID(left_ranges, right_ranges)
       
        return self.SPEED, steering_angle