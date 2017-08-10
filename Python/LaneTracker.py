import cv2
import numpy as np
from helpers import LineCluster


class LaneTracker:
    """
    Class to track lanes over time in image
    """

    def __init__(self, min_y, max_y):
        """
        Initialize new LaneTracker object
        @param min_y: minimum y value of detected lanes
        @type min_y: int
        @param max_y: maximum y value of detected lanes
        @type max_y: int
        """
        self.min_y = min_y
        self.max_y = max_y
        self.left_x = None
        self.right_x = None
        self.left_velocity = [0, 0]
        self.right_velocity = [0, 0]
        self.out_right = [False, False]
        self.out_left = [False, False]

    def left_line(self):
        """
        Returns left side of the tracked lane
        @return: Line indicating the left of the lane
        @rtype: ((int, int), (int, int))
        """
        return (self.left_x[0], self.min_y), (self.left_x[1], self.max_y)

    def right_line(self):
        """
        Returns right side of the tracked lane
        @return: Line indicating the right of the lane
        @rtype: ((int, int), (int, int))
        """
        return (self.right_x[0], self.min_y), (self.right_x[1], self.max_y)

    def polygon(self):
        """
        Returns a polygon indicating the tracked lane
        @return: Four points indicating polygon's vertices
        @rtype: ((int, int), (int, int), (int, int), (int, int))
        """
        return np.array([[self.left_x[0], self.min_y], [self.left_x[1], self.max_y],
                         [self.right_x[1], self.max_y], [self.right_x[0], self.min_y]], np.int32)

    def step(self, image):
        """
        Updates the tracked lane with the next frame in the source video
        @param image: Next frame of the video
        @type image: Any
        @return: None
        """
        x_mid = np.int(np.float(image.shape[1])/2)

        # Find the edges of the image
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        canny_image = cv2.Canny(blurred_image, 50, 150)

        # Add a mask to ignore non-track lines
        vertices = np.array([[(0, canny_image.shape[0]),
                              (0, canny_image.shape[0] * 0.6),
                              # (canny_image.shape[1] * 0.2, canny_image.shape[0] * 0.4),
                              # (canny_image.shape[1] * 0.8, canny_image.shape[0] * 0.4),
                              (canny_image.shape[1], canny_image.shape[0] * 0.6),
                              (canny_image.shape[1], canny_image.shape[0])]],
                            dtype=np.int32)
        mask = np.zeros_like(canny_image)
        if len(canny_image.shape) > 2:
            channel_count = canny_image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(canny_image, mask)

        # Find line clusters in image
        lines = cv2.HoughLinesP(masked_image, 1, np.pi/200, 4, minLineLength=self.min_y*0.2, maxLineGap=self.min_y*0.05)
        if lines is None:
            # TODO handle no lines being found in image
            return None
        clusters = []
        for x1, y1, x2, y2 in lines[0]:
            if y2-y1 == 0:
                continue  # Ignore horizontal lines
            if x2-x1 == 0:
                x2 -= 1  # Change vertical lines for simplicity

            check = False
            for cluster in clusters:
                if cluster.check_append((x1, y1), (x2, y2)):
                    check = True
                    break
            if not check:
                new_cluster = LineCluster(self.min_y, self.max_y, (x1, y1), (x2, y2))
                clusters.append(new_cluster)

        # Find the lane lines to the right and left of the runner
        if self.right_x is None:  # If no corners are set, set them all
            r_line, rx = None, float("Inf")
            l_line, lx = None, float("-Inf")
            for line in clusters:
                x_dist = (x_mid - line.bottom_x()) + (x_mid - line.top_x())
                if rx > x_dist > 0:
                    r_line, rx = line, x_dist
                elif 0 > x_dist > lx:
                    l_line, lx = line, x_dist
            self.right_x = [r_line.bottom_x(), r_line.top_x()]
            self.left_x = [l_line.bottom_x(), l_line.top_x()]

        else:
            r_line, r_dist = None, 5000
            l_line, l_dist = None, 5000
            r_side, r_side_dist = None, float("Inf")
            l_side, l_side_dist = None, float("Inf")
            for line in clusters:
                r_error = (np.float(np.square(abs(self.right_x[0] - line.bottom_x())) +
                                    np.square(abs(self.right_x[1] - line.top_x()))))/2
                l_error = (np.float(np.square(abs(self.left_x[0] - line.bottom_x())) +
                                    np.square(abs(self.left_x[1] - line.top_x()))))/2
                r_dist = (np.float(np.square(abs(image.shape[0] - line.bottom_x())) +
                                   np.square(abs(image.shape[0] - line.top_x()))))/2
                l_dist = np.float(np.square(line.bottom_x()) + np.square(line.top_x()))/2

                if r_error < r_dist:
                    r_line, r_dist = line, r_error
                elif l_error < l_dist:
                    l_line, l_dist = line, l_error
                elif r_dist < r_side_dist:
                    r_side, r_side_dist = line, r_dist
                elif l_dist < l_side_dist:
                    l_side, l_side_dist = line, l_dist

            if r_line is not None:
                self.right_velocity = [self.right_x[0] - r_line.bottom_x(), self.right_x[1] - r_line.top_x()]
                self.right_x = [r_line.bottom_x(), r_line.top_x()]

            if l_line is not None:
                self.left_velocity = [self.left_x[0] - l_line.bottom_x(), self.left_x[1] - l_line.top_x()]
                self.left_x = [l_line.bottom_x(), l_line.top_x()]

                # if r_line is None:
                #     if not (self.out_right[0] or self.out_right[1]):
                #         prev_dist_r = (np.float(np.square(abs(image.shape[0] - self.right_x[0])) +
                #                                 np.square(abs(image.shape[0] - self.right_x[1]))))/2
                #         if prev_dist_r < 5000:
                #             print "Dead End"

                # if r_line is None: # Right line not found
                #     if not (self.out_right[0] or self.out_right[1]): # It was not already off screen
                #         prev_dist_r = (np.float(np.square(abs(image.shape[0] - self.right_x[0])) +
                #                               np.square(abs(image.shape[0] - self.right_x[1]))))/2
                #         prev_dist_l = (np.float(np.square(self.right_x[0]) +
                #                               np.square(self.right_x[1])))/2
                #         if prev_dist_r < 5000:
                #             self.out_right = [False, True]
                #         if prev_dist_l < 5000:
                #             self.out_right = [True, False]

            if r_line is None and l_line is not None:
                self.right_x = [self.right_x[0] - self.left_velocity[0], self.right_x[1] - self.left_velocity[1]]
            if l_line is None and r_line is not None:
                self.left_x = [self.left_x[0] - self.right_velocity[0], self.left_x[1] - self.right_velocity[1]]
