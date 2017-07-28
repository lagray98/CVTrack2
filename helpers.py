import cv2
import numpy as np
import random
from matplotlib import pyplot as plt


class LinedImage:
    """Class to add lines to an image"""
    def __init__(self, image):
        self.image = image
        self.edged_image = None
        self.lined_image = None
        self.just_lines = None
        self.canny = [100, 250, 3]
        self.hough = [1, np.pi/180, 200]
        self.lines = []

    def set_canny(self, threshold1, threshold2, aperture_size=3):
        """Set the Parameters for the Canny Edge Detection"""
        self.canny = [threshold1, threshold2, aperture_size]

    def set_hough(self, rho, theta, threshold):
        """Set the parameters for the Hough Line Transform"""
        self.hough = [rho, theta, threshold]

    def run_avg(self, min_slope=50):
        """Run a Canny Edge Detection followed by a Hough Line Transform on the base image to create three images:
            self.edged_image: cv2 image with just Canny Edge Detection
            self.lined_image: base image with the lines from the Hough Line Transform overlaid
            self.just_lines: cv2 image of just the lines
        """
        self.edged_image = cv2.Canny(self.image, self.canny[0], self.canny[1], apertureSize=self.canny[2])
        self.just_lines = np.ones(self.image.shape, np.uint8)
        self.lined_image = self.image.copy()
        # self.lines = cv2.HoughLines(self.edged_image, 1, np.pi / 180, 200)
        self.lines = cv2.HoughLinesP(self.edged_image, self.hough[0], self.hough[1], self.hough[2],
                                     minLineLength=self.image.shape[0]/3, maxLineGap=10)[0]
        self.lines = [line for line in self.lines if abs(line[3]-line[1]) >= min_slope]
        avgslope = 0
        for x1,y1,x2,y2 in self.lines:
            avgslope += (y2-y1)/(x2-x1)
        avgslope = avgslope / len(self.lines)
        lx1,lx2, rx1, rx2 = [], [], [], []
        y_min = self.image.shape[0]
        y_max = int(self.image.shape[0] * 0.611)
        for x1, y1, x2, y2 in self.lines:
            if (y2-y1)/(x2-x1) < avgslope:
                mc = np.polyfit([x1,x2], [y1, y2], 1)
                lx1.append(np.int(np.float((y_min - mc[1]))/np.float(mc[0])))
                lx2.append(np.int(np.float((y_max - mc[1]))/np.float(mc[0])))
            elif (y2-y1)/(x2-x1) > avgslope:
                mc = np.polyfit([x1,x2],[y1,y2], 1)
                rx1.append(np.int(np.float((y_min - mc[1])) / np.float(mc[0])))
                rx2.append(np.int(np.float((y_max - mc[1])) / np.float(mc[0])))
        lx1_avg = np.int(np.nanmean(lx1))
        lx2_avg = np.int(np.nanmean(lx2))
        rx1_avg = np.int(np.nanmean(rx1))
        rx2_avg = np.int(np.nanmean(rx2))
        midx1_avg = (lx1_avg+rx1_avg) / 2
        midx2_avg = (lx2_avg+rx2_avg) / 2

        cv2.line(self.lined_image, (lx1_avg, y_min), (lx2_avg, y_max), (0, 0, 255), 2)
        cv2.line(self.just_lines, (lx1_avg, y_min), (lx2_avg, y_max), (0, 0, 255), 2)
        cv2.line(self.lined_image, (rx1_avg, y_min), (rx2_avg, y_max), (0, 0, 255), 2)
        cv2.line(self.just_lines, (rx1_avg, y_min), (rx2_avg, y_max), (0, 0, 255), 2)
        cv2.line(self.lined_image, (midx1_avg, y_min), (midx2_avg, y_max), (0, 0, 255), 4)
        cv2.line(self.just_lines, (midx1_avg, y_min), (midx2_avg, y_max), (0, 0, 255), 4)

    def run(self, min_slope=50):
        """Run a Canny Edge Detection followed by a Hough Line Transform on the base image to create three images:
            self.edged_image: cv2 image with just Canny Edge Detection
            self.lined_image: base image with the lines from the Hough Line Transform overlaid
            self.just_lines: cv2 image of just the lines
        """
        self.edged_image = cv2.Canny(self.image, self.canny[0], self.canny[1], apertureSize=self.canny[2])
        self.just_lines = np.ones(self.image.shape, np.uint8)
        self.lined_image = self.image.copy()
        # self.lines = cv2.HoughLines(self.edged_image, 1, np.pi / 180, 200)
        self.lines = cv2.HoughLinesP(self.edged_image, self.hough[0], self.hough[1], self.hough[2],
                                     minLineLength=self.image.shape[0]/3, maxLineGap=10)[0]
        for x1, y1, x2, y2 in self.lines:
            if abs(y2-y1) < min_slope: continue
            cv2.line(self.lined_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(self.just_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    def get_lines(self):
        self.edged_image = cv2.Canny(self.image, self.canny[0], self.canny[1], apertureSize=self.canny[2])
        # self.lines = cv2.HoughLines(self.edged_image, 1, np.pi / 180, 200)
        self.lines = cv2.HoughLinesP(self.edged_image, self.hough[0], self.hough[1], self.hough[2],
                                     minLineLength=self.image.shape[0]/3, maxLineGap=10)[0]
        avgslope = 0
        for x1,y1,x2,y2 in self.lines:
            if abs(y2-y1) < 50: continue
            avgslope += (y2-y1)/(x2-x1)
        avgslope = avgslope / len(self.lines)
        lx1,lx2, rx1, rx2 = [], [], [], []
        y_min = self.image.shape[0]
        y_max = int(self.image.shape[0] * 0.611)
        for x1, y1, x2, y2 in self.lines:
            if (y2-y1)/(x2-x1) < avgslope:
                mc = np.polyfit([x1,x2], [y1, y2], 1)
                lx1.append(np.int(np.float((y_min - mc[1]))/np.float(mc[0])))
                lx2.append(np.int(np.float((y_max - mc[1]))/np.float(mc[0])))
            elif (y2-y1)/(x2-x1) > avgslope:
                mc = np.polyfit([x1,x2],[y1,y2], 1)
                rx1.append(np.int(np.float((y_min - mc[1])) / np.float(mc[0])))
                rx2.append(np.int(np.float((y_max - mc[1])) / np.float(mc[0])))
        lx1_avg = np.int(np.nanmean(lx1))
        lx2_avg = np.int(np.nanmean(lx2))
        rx1_avg = np.int(np.nanmean(rx1))
        rx2_avg = np.int(np.nanmean(rx2))
        midx1_avg = (lx1_avg+rx1_avg) / 2
        midx2_avg = (lx2_avg+rx2_avg) / 2
        return [(lx1_avg, y_min), (lx2_avg, y_max)], [(rx1_avg, y_min), (rx2_avg, y_max)], [(midx1_avg, y_min), (midx2_avg, y_max)]

    def display(self, only_lines=True):
        """Displays either the original image, Canny Edge image, and Hough Line image or
         just the Hough Line image and Hough lines on a white background using matplotlib"""
        if not only_lines:
            plt.subplot(3, 1, 1), plt.imshow(self.image, cmap='gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(3, 1, 2), plt.imshow(self.edged_image, cmap='gray')
            plt.title('Edged Image'), plt.xticks([]), plt.yticks([])
            plt.title('Lined Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(3, 1, 3), plt.imshow(self.lined_image, cmap='gray')
        else:
            plt.subplot(1, 2, 1), plt.imshow(self.just_lines, cmap='gray')
            plt.title('Lines Only'), plt.xticks([]), plt.yticks([])
            plt.subplot(1, 2, 2), plt.imshow(self.lined_image, cmap='gray')
            plt.title('Lined Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def save(self, name):
        cv2.imwrite('output/'+name+'_edges.png', self.edged_image)
        cv2.imwrite('output/'+name+'_lines.png', self.lined_image)
