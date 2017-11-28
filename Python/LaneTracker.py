import cv2
import numpy as np
from helpers import LineCluster
from detect_color import *
from helpers import findAngle



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
                         [self.right_x[1], self.max_y], [self.right_x[0], self.min_y]], np.int32) #converts to int array
    
    

    def step(self, image):
        """
        Updates the tracked lane with the next frame in the source video
        @param image: Next frame of the video
        @type image: Any
        @return: None
        """
        x_mid = np.int(np.float(image.shape[1])/2) #finds center of image
        

        # Find the edges of the image
        
        color_masked_image = detectColor(image)
        
        cv2.imshow('colorMask',color_masked_image)
#        cv2.waitKey(0)

        blurred_image = cv2.GaussianBlur(color_masked_image, (5, 5), 0)
        canny_image = cv2.Canny(blurred_image, 50, 150) #<type 'numpy.ndarray'>
        

        # Add a mask to ignore non-track lines, sky
#        vertices = np.array([[(0, canny_image.shape[0]),
#                              (0, canny_image.shape[0] * 0.35),
#                              # (canny_image.shape[1] * 0.2, canny_image.shape[0] * 0.4),
#                              # (canny_image.shape[1] * 0.8, canny_image.shape[0] * 0.4),
#                              (canny_image.shape[1], canny_image.shape[0] * 0.35),
#                              (canny_image.shape[1], canny_image.shape[0])]],
#                            dtype=np.int32)
#        mask = np.zeros_like(canny_image)
#        
#        # mask different if black and white vs. color
#        if len(canny_image.shape) > 2:
#            channel_count = canny_image.shape[2]
#            ignore_mask_color = (255,) * channel_count
#        else:
#            ignore_mask_color = 255
#        cv2.fillPoly(mask, vertices, ignore_mask_color)
#        masked_image = cv2.bitwise_and(canny_image, mask) #<type 'numpy.ndarray'>


        cv2.imshow('masked',canny_image)


#        cv2.waitKey(0)

        

        # Find line clusters in image
        # Line length must be at least 20% of image
        # Max gap between 2 lines if 5% of image
        lines = cv2.HoughLinesP(canny_image, 1, np.pi/200, 4, minLineLength=self.min_y*0.35, maxLineGap=self.min_y*0.05) #Optimal for red mask: minLen= y*.25, maxGap = y*.05, black mask min len = 0.35
        if lines is None:
            # TODO handle no lines being found in image
            return None
        clusters = []
        for x1, y1, x2, y2 in lines[0]:
            if y2-y1 == 0:
                continue  # Ignore horizontal lines
            if x2-x1 == 0:
                x2 -= 1  # Change vertical lines for simplicity (no division by zero)

            check = False
            
            for cluster in clusters:
                if cluster.check_append((x1, y1), (x2, y2)):
                    check = True
                    break
            if not check:
                new_cluster = LineCluster(self.min_y, self.max_y, (x1, y1), (x2, y2))
                clusters.append(new_cluster)
            
            cv2.line(image,(x1,y1),(x2,y2), (255,0,0),10)
                
        cv2.imshow('track',image)
#        cv2.waitKey(0)




        # Find the lane lines to the right and left of the runner, l_line and r_line ?
#        if self.right_x is None:  # If no corners are set, set them all
#            r_line, rx = None, float("-Inf")
#            l_line, lx = None, float("Inf")
#            for line in clusters: # Determines if line is the left or right line depending on if x_dist is + or -, switch so only bottom_x
#                x_dist = (x_mid - line.bottom_x()) + (x_mid - line.top_x())
#                if rx <= x_dist < 0:
#                    r_line, rx = line, x_dist
#                elif 0 < x_dist <= lx:
#                    l_line, lx = line, x_dist
#            self.right_x = [r_line.bottom_x(), r_line.top_x()]
#            self.left_x = [l_line.bottom_x(), l_line.top_x()]

        if self.right_x is None:
            r_line, rx = None, float("-Inf")
            l_line, lx = None, float("Inf")
#            for line in clusters:
#                x_dist = x_mid - line.bottom_x()
#                if rx <= x_dist < 0:
#                    r_line, rx = line, x_dist
#                elif 0 < x_dist <= lx:
#                    l_line, lx = line, x_dist
#            self.right_x = [r_line.bottom_x(), r_line.top_x()]
#            self.left_x = [l_line.bottom_x(), l_line.top_x()]


            rightLines = []
            leftLines = []
            
            for line in clusters:
#                print("Line top x: " + str(line.top_x()))
#                print("Line bottom x: " + str(line.bottom_x()))

                x_dist = x_mid - line.bottom_x()
#                print("X distance: " + str(x_dist) + "\n")
                if x_dist < 0:
                    rightLines.append(line)
                elif 0 < x_dist:
                    leftLines.append(line)
            
            for rightLine in rightLines:
                for leftLine in leftLines:
                    if (50 < findAngle([leftLine.bottom_x(),leftLine.top_x()],[rightLine.bottom_x(),rightLine.top_x()],self.min_y) < 60):
                        r_line = rightLine
                        l_line = leftLine
                        self.right_x = [rightLine.bottom_x(), rightLine.top_x()]
                        self.left_x = [leftLine.bottom_x(),leftLine.top_x()]


            cv2.line(image,(r_line.bottom_x(), int(self.min_y)), (r_line.top_x(), int(self.max_y)),(0,0,255),5)
            cv2.line(image,(l_line.bottom_x(), int(self.min_y)), (l_line.top_x(), int(self.max_y)),(0,0,255),5)
            
            print("Right bottom x: " + str(r_line.bottom_x()))
            print("Left bottom x: " + str(l_line.bottom_x()))

            cv2.imshow('track',image)
#            cv2.waitKey(0)


        else:
#            r_line, r_dist = None, 5000
#            l_line, l_dist = None, 5000
#            r_side, r_side_dist = None, float("Inf") #infinite value
#            l_side, l_side_dist = None, float("Inf")
#            for line in clusters: # checks for 'error'
#                r_error = (np.float(np.sqrt(np.square(abs(self.right_x[0] - line.bottom_x())) +  #take square root instead of dividing by two
#                                    np.square(abs(self.right_x[1] - line.top_x())))))
#                l_error = (np.float(np.sqrt(np.square(abs(self.left_x[0] - line.bottom_x())) +
#                                    np.square(abs(self.left_x[1] - line.top_x())))))
#                r_dist = (np.float(np.sqrt(np.square(abs(image.shape[0] - line.bottom_x())) +
#                                           np.square(abs(image.shape[0] - line.top_x()))))) #image.shape[1]?
#                l_dist = np.float(np.sqrt(np.square(line.bottom_x()) + np.square(line.top_x())))
#
#                if r_error < r_dist:
#                    r_line, r_dist = line, r_error
#                elif l_error < l_dist:
#                    l_line, l_dist = line, l_error
#                elif r_dist < r_side_dist:
#                    r_side, r_side_dist = line, r_dist #r_side and l_side not used after this ?
#                elif l_dist < l_side_dist:
#                    l_side, l_side_dist = line, l_dist

            r_line, rx = None, float("-Inf")
            l_line, lx = None, float("Inf")
            
#            for line in clusters:
#                print("Line top x: " + str(line.top_x()))
#                print("Line bottom x: " + str(line.bottom_x()))
#
#                x_dist = x_mid - line.bottom_x()
#                print("X distance: " + str(x_dist) + "\n")
#                if rx <= x_dist < 0:
#                    r_line, rx = line, x_dist
#                elif 0 < x_dist <= lx:
#                    l_line, lx = line, x_dist


            rightLines = []
            leftLines = []
        
            for line in clusters:
#                print("Line top x: " + str(line.top_x()))
#                print("Line bottom x: " + str(line.bottom_x()))

                x_dist = x_mid - line.bottom_x()
#                print("X distance: " + str(x_dist) + "\n")
                if x_dist < 0:
                    rightLines.append(line)
                elif 0 < x_dist:
                    leftLines.append(line)
                    
            for rightLine in rightLines:
                for leftLine in leftLines:
                    if (50 < findAngle([leftLine.bottom_x(),leftLine.top_x()],[rightLine.bottom_x(),rightLine.top_x()],self.min_y) < 60):
                        r_line = rightLine
                        l_line = leftLine


            if r_line is not None and l_line is not None:
                self.right_x = [r_line.bottom_x(), r_line.top_x()]
                self.left_x = [l_line.bottom_x(), l_line.top_x()]



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


            cv2.line(image,(self.right_x[0], int(self.min_y)), (self.right_x[1], int(self.max_y)),(0,0,255),5)
            cv2.line(image,(self.left_x[0], int(self.min_y)), (self.left_x[1], int(self.max_y)),(0,0,255),5)
            
#            print("Right bottom x: " + str(self.right_x[0]))
#            print("Left bottom x: " + str(self.left_x[0]))
            cv2.imshow('track',image)
#            cv2.waitKey(0)

        if(self.right_x[0] < 110 or self.left_x[0] > 35):
            print("VEERING")
            return 1
        return 0
#        topAngle = findAngle(self.left_x,self.right_x, self.min_y)


#leftSlope = float(self.min_y - self.max_y) / float(self.left_x[1] - self.left_x[0])
#print("Left Slope: " + str(leftSlope))

#rightSlope = float(self.min_y - self.max_y) / float(self.right_x[1] - self.right_x[0])
#print("Right Slope: " + str(rightSlope))

