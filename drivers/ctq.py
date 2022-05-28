import numpy as np
import time

from numpy.lib import gradient

class CTQmk4:    

    BUBBLE_RADIUS = 160
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 120
    MAX_LIDAR_DIST = 10000000
    BASE_SPEED = 6.5
    STRAIGHTS_STEERING_ANGLE = np.pi / 18  # 10 degrees
    SEMI_CLOSE_THRESHOLD = 5
    CLOSE_THRESHOLD = 3
    TURBO_TIMER = 5
    
    def __init__(self):
        # Used to create turbo start
        self.start_counter = 0
        self.turbo_speed = 100*np.sin(np.exp(2*np.pi*((-0.1*(self.start_counter)) / 360))) + 45
        # Used to calculate steering_angle
        self.radians_per_elem = None
        # Used to calculate racing variables
        self.best_speed = self.BASE_SPEED
        self.on_straight = False
        self.last_curve_coeff = 0
        self.last_cornering_coeff = self.BEST_POINT_CONV_SIZE
        # Visualiser Data
        self.visualiser_range = []
        self.best_point = 0
        self.gradients_mean = 0
    
    def set_bounds_i(self, array, index, r, i):
        """
        Multiplies values within a radius r of the index with i 
        """
        min_index = index - r
        max_index = index + r
        if min_index < 0: min_index = 0
        if max_index >= len(array): max_index = len(array) - 1

        tag = array[min_index:max_index]
        tag_fixed = list(map(lambda x: x*i, tag))
        array[min_index:max_index] = tag_fixed
        return array

    def smooth_change(self, current, last, max_change):
        """
        Limits the changes of a variable to "max_change" per frame
        """
        if abs(current - last) > max_change:
        #if corning_coeff is increasing
            if current > last:
                current = last + max_change
            else: 
                current = last - max_change
        return current

    def find_race_coeff(self, array, index, gap_start, gap_end):
        """
        Calculates the curve coefficient of the next turn as well
        as the automatic emergency braking coefficient if necessary
        """
        # calculate curve_coeff
        min_index = index - 10
        max_index = index + 10
        if min_index < gap_start:
            min_index = gap_start
        if max_index > gap_end:
            max_index = gap_end
        curve_gradients = np.gradient(array[min_index:max_index])
        curve_mean = abs(np.mean(curve_gradients))
        # Typical curve_mean values :
            #A sharp corner is about (0.01 - 0.05)
            #A wide corner is about (0.05 - 0.08)
        curve_coeff = 1.8 * (np.power(curve_mean, (1.0/7.0))) # 1.6 is taking 0.04 as average velocity

        # Straight Detection
        high = np.argwhere(array > 25) # Max Lidar distance is about 30, see self.MAX_LIDAR_DIST
        if len(high) > 20:
            self.on_straight = True
        else:
            self.on_straight = False
        if curve_coeff > 1.2 or self.on_straight: curve_coeff = 1.2
        if curve_coeff < 0.75: curve_coeff = 0.5
        self.last_curve_coeff = curve_coeff

        # Calculate aeb_coeff
        aeb_range = array[248:572] # 324 Values in centre
        aeb_coeff = 1
        sceneario1_est = len(np.argwhere(aeb_range < self.CLOSE_THRESHOLD))
        sceneario2_est = len(np.argwhere(aeb_range < self. SEMI_CLOSE_THRESHOLD))
        if sceneario2_est > 250:
            if sceneario1_est > 300:
                aeb_coeff = 0.1
                #print("AEB sceneario 1")
            else: 
                aeb_coeff = aeb_range.mean() / self.CLOSE_THRESHOLD
                #print("AEB sceneario 2")

        return curve_coeff, aeb_coeff

    def optimise_speed(self, distance, base_speed):
        """
        Calculates the optimised speed according to polynomial
        """
        opt_speed = base_speed + 0.04*(distance**1.5) # Change polynomial to suit race
        return opt_speed

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window (via convolution)
            2.Rejecting high values (eg. > 3m)
            Also alters some values to guide the steering system to the best point
        """
        self.radians_per_elem = (2*np.pi) / len(ranges)
	    # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges[135:-135])

        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)

        # Finds closest point and set to zero
        closest = np.argmin(proc_ranges)       
        # Find points that are "too_close" and set to i ( useful for multi-agent racing )
        too_close_range = proc_ranges[248:572] # 324 Values in centre ( AEB range )
        too_close = np.argwhere(too_close_range < self.CLOSE_THRESHOLD).flatten()
        for close in too_close:
            proc_ranges[close] = proc_ranges[close]*0.5
        proc_ranges = self.set_bounds_i(proc_ranges, closest, self.BUBBLE_RADIUS, 0)

        return proc_ranges

    def find_gradients(self, proc_ranges):
        """
        Finds the mean gradient of the track.
        Indicates the presence of obstacles and other cars on the track.
        """
        gradients_raw = list(map(lambda x: abs(x), np.gradient(proc_ranges)))
        gradients_raw = list(map(lambda x: abs(x), np.gradient(gradients_raw)))
        gradients = np.convolve(gradients_raw, np.ones_like(proc_ranges), 'same')
        self.gradients_mean = np.mean(gradients)

    def slice_score(self, proc_ranges, slice):
        """
        Scores a slice according to it's size and mean values
        """
        gap_start = slice.start
        gap_end = slice.stop
        gap_size = gap_end - gap_start
        width_score = (gap_size / 810)
        distance_score = proc_ranges[gap_start:gap_end].mean() / proc_ranges.mean()
        slice_score = (width_score*0.7) + (distance_score*0.3)
        return slice_score

    def find_best_point_mk2(self, proc_ranges):
        """
        Uses self.slice_score() system to determine the best gap to follow
        """
        # mask the bubble
        masked = np.ma.masked_where(proc_ranges==0, proc_ranges)
        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        if len(slices) <= 1:
            gap_start = 0
            gap_end = len(proc_ranges) - 1
            best = self.find_best_point(gap_start, gap_end, proc_ranges)
            return best, gap_start, gap_end
        else:
            # self.slice_score() takes in gap size and distances of slice and produces a "slice_score"
            slice_scores = list(map(lambda x: self.slice_score(proc_ranges, x), slices))
            chosen_slice = slices[np.argmax(slice_scores)]
            gap_start = chosen_slice.start
            gap_end = chosen_slice.stop
            best = self.find_best_point(gap_start, gap_end, proc_ranges)
            return best, gap_start, gap_end
    
    def find_best_point(self, start_i, end_i, ranges):
        """
        Takes in gap and chooses the best point to travel towards
        cornerning_coeff determines how "wide" to take the corners
            governed by the mean gradient of the lidar data and the curve of the barrier ahead
        If on a straight, the 
        """
        if self.gradients_mean > 20:
            cornering_coeff = ((self.gradients_mean / 20)**1.3) * self.BEST_POINT_CONV_SIZE * ((1 / self.last_curve_coeff)**1.3)
        else:
            cornering_coeff = self.BEST_POINT_CONV_SIZE

        #smooth cornering coeff
        cornering_coeff = self.smooth_change(cornering_coeff, self.last_cornering_coeff, 20)
        self.last_cornering_coeff = cornering_coeff
        
        if self.on_straight:
            masked = np.ma.masked_where(ranges<25, ranges)
            # get a slice for each contigous sequence of non-bubble data
            straight_slices = np.ma.notmasked_contiguous(masked)
            if len(straight_slices) > 0:
                chosen_slice = straight_slices[0]
                chosen_slice_score = self.slice_score(ranges, chosen_slice)
                for slice in straight_slices:
                    slice_score = self.slice_score(ranges, slice)
                    if slice_score > chosen_slice_score:
                        chosen_slice = slice
                        chosen_slice_score = slice_score
                    gap_start, gap_end = chosen_slice.start, chosen_slice.stop
                averaged_max_gap = np.convolve(ranges[gap_start:gap_end], np.ones(int(cornering_coeff)), 'same') / cornering_coeff
                return averaged_max_gap.argmax() + gap_start
            else: 
                self.on_straight = False
        if not self.on_straight:
            averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(int(cornering_coeff)), 'same') / cornering_coeff
            return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the LiDAR data and transform it into an appropriate steering angle
        """
        lidar_angle = (range_index - (range_len/2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
        return steering_angle

    def process_lidar(self, ranges):
        """ Process each LiDAR scan and output a speed and steering_angle
        """
        self.visualiser_range = ranges
        proc_ranges = self.preprocess_lidar(ranges)
    
        #Finds best point to travel to
        best, gap_start, gap_end = self.find_best_point_mk2(proc_ranges)
        self.best_point = best

        # Find lidar gradient
        self.find_gradients(proc_ranges)

        #Find speed_coeff (speed_coeff = curve_coeff * aeb_coeff)
        curve_coeff, aeb_coeff = self.find_race_coeff(proc_ranges, best, gap_start, gap_end)

        #Optimise Speed
        best_speed = self.optimise_speed(proc_ranges[best], self.BASE_SPEED)
        if self.start_counter > 50:
            curve_coeff = self.smooth_change(curve_coeff, self.last_curve_coeff, 0.02)
            best_speed *= curve_coeff

        #Publish Drive message
        steering_angle = self.get_angle(best, len(proc_ranges))
        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
            speed = best_speed * 0.8
        else: 
            speed = best_speed
        speed *= aeb_coeff
        speed = self.smooth_change(speed, self.best_speed, 0.05)
        if self.start_counter < self.TURBO_TIMER:
            speed = self.turbo_speed
        else:
            if speed > 25:
                speed = 25
        if self.start_counter < self.TURBO_TIMER:
            self.start_counter += 1
        self.best_speed = speed
        
        print(f"Speed:{speed}, Steering Angle:{steering_angle}")
        return speed, steering_angle

    def get_visualiser_ranges(self):
        """
        Outputs visualiser data
        """
        return self.visualiser_range, self.best_point, self.best_speed