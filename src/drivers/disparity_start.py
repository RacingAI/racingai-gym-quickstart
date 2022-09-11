"""
These imports aren't required to impliment the algorithm.
Everything can be accomplished in base python. Numpy provides
some array functions that you may find useful. It might also
allow you to write some faster code. Matplotlib is a visualisation
library. You are NOT required to create any visualizations but it
may help you see what your algorithm is doing
"""

import numpy as np
import matplotlib.pyplot as plt


class DisparityExtender:
    def __init__(self) -> None:
        """
        The variables you need are defined here. Play with some of them
        and see what changes. If you need to declare any more varibles
        feel free to do so
        """
        self.CAR_WIDTH = 0.31           # Keep this variable constant. DO NOT CHANGE
        self.DIFFERENCE_THRESHOLD = 2.  # Play with this variable and see what changes
        self.SPEED = 5                  # You will be implimenting your own speed function to change this
        self.SAFETY_PERCENTAGE = 300    # Play with this variable and see what changes

    def preprocess_lidar(self, ranges):
        """
        This is where you have to limit your FOV
        """
    
    def get_differences(self, ranges):
        """
        Calculate the differences between adjacent LiDAR scans. They 
        should NOT be negative. Return as an array
        """
    
    def get_disparities(self, differecnes):
        """
        Find the indices where the disparities are. Return as an array.
        """
    
    def get_num_points_to_cover(self, distance, width):
        """
        Given the distance of the closest LiDAR scan, you need to
        figure out how many lidar points you need to cover half the
        width of the car
        """
    
    def cover_points(self, num_points, start_idx, cover_right, ranges):
        """
        Covers the number of LiDAR points returned by get_num_points_to_cover()
        with the closer distance, starting from start_idx.
        start_idx is the index of the closer distance
        cover_right is a bool that determines whether you cover left to right (True)
            or right to left (False)    
        """
    
    def extend_disparities(self, disparities, ranges, car_width, extra_percentage):
        """
        This function uses the above two functions to extend the disparities
        as mentioned in the PowerPoint. The handout will be useful if you need
        a refresher
        """

    def get_steering_angle(self, ranges):
        """
        Choose the longest LiDAR scan and fiure out the associated steering
        angle.
        """

    def get_speed(self, ranges):
        """
        Here is where you are allowed full creativity. Impress us with your own
        implimenation of a speed function. The easiest function would be to set
        it to a constant ie:
        self.SPEED = 5
        """

    def run_algorithm(self, ranges):
        """
        Here is where you use the functions you implimented above to actually
        build out and run the algorithm.
        IMPORTANT: Return the speed first and then the steering angle. The last
        line should be:
            return speed, steering_angle

        (Or whatever you named those variables. It is just important that
        the speed comes first and then the steering angle)
        """

        return speed, steering_angle

