import cv2
import numpy as np
import random
from matplotlib import pyplot as plt


def stabilize(file_name):
    cap = cv2.VideoCapture(file_name)
    # output_video = cv2.VideoWriter('output_py.avi', cv2.cv.CV_FOURCC(*'SVQ3'), 10.0, (prev.shape[1], prev.shape[0]))
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=200,
                          qualityLevel=0.1,
                          minDistance=20,
                          blockSize=10)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    ret, frame = cap.read()
    shift_x = 0
    shift_y = 0
    while (ret):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Get the average amount of shift
        x = []
        y = []
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            x.append(c-a)
            y.append(d-b)
            cv2.circle(frame, (a,b), 3, 255, -1)
        shift_x -= np.mean(x)
        shift_y -= np.mean(y)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        rows, cols, _ = frame.shape
        output_frame = cv2.warpAffine(frame, M, dsize=(rows, cols))

        cv2.imshow('frame', output_frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        # p0 = good_new.reshape(-1, 1, 2)
        ret, frame = cap.read()


    cv2.destroyAllWindows()
    cap.release()


def canny_mask(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.Canny(img, 50, 150)

    # defining a blank mask to start with
    vertices = np.array([[(0, img.shape[0]),
                          (0, img.shape[0]*0.6),
                          (img.shape[1]*0.2, img.shape[0]*0.4),
                          (img.shape[1]*0.8, img.shape[0]*0.4),
                          (img.shape[1], img.shape[0]*0.6),
                          (img.shape[1], img.shape[0])]],
                        dtype=np.int32)
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


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
        y_min = self.image.shape[0]
        y_max = int(self.image.shape[0] * 0.611)
        x_mid = int(self.image.shape[1] / 2)

        # Find the two line clusters on either side of the center
        best_l, slope_l = 1000, 0
        best_r, slope_r = 1000, 0
        for x1, y1, x2, y2 in self.lines:
            slope = (float(y2)-y1)/(x2-x1)
            if y_min-y1 < 10 and x_mid-x1 < best_l:
                slope_l, best_l = slope, x_mid-x1
            if y_min-y2 < 10 and x2-x_mid < best_r:
                slope_r, best_r = slope, x2-x_mid

        # Get the average lines of the two clusters
        lx1, lx2, rx1, rx2 = [], [], [], []
        for x1, y1, x2, y2 in self.lines:
            if abs((float(y2)-y1)/(x2-x1) - slope_l) < 0.5:
                mc = np.polyfit([x1, x2], [y1, y2], 1)
                lx1.append(np.int(np.float((y_min - mc[1]))/np.float(mc[0])))
                lx2.append(np.int(np.float((y_max - mc[1]))/np.float(mc[0])))
            elif abs((float(y2) - y1) / (x2 - x1) - slope_r) < 0.5:
                mc = np.polyfit([x1, x2], [y1, y2], 1)
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
            color = random.randint(0,255)
            cv2.line(self.lined_image, (x1, y1), (x2, y2), (color, color, color), 2)
            cv2.line(self.just_lines, (x1, y1), (x2, y2), (color, color, color), 2)

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


def track_lanes(image):
    y_min = image.shape[0]
    y_max = int(image.shape[0] * 0.4)
    x_mid = int(image.shape[1] / 2)
    canny = canny_mask(image)
    lines = cv2.HoughLinesP(canny, 1, np.pi/200, 4,
                            minLineLength=y_min*0.2, maxLineGap=y_min*0.05)
    if lines is None:
        return image
    lines = lines[0]
    # for x1, y1, x2, y2 in lines:
    #     if abs(y2 - y1) < 50: continue
    #     cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 5)
    # return image
    best_l, slope_l = 1000, 0
    best_r, slope_r = 1000, 0
    for x1, y1, x2, y2 in lines:
        slope = (float(y2) - y1) / (x2 - x1)
        print slope
        if abs(slope) <= 1.5:
            continue
        if y_min - y1 < y_min*0.3 and x_mid - x1 < best_l:
            slope_l, best_l = slope, x_mid - x1
        if y_min - y2 < y_min*0.3 and x2 - x_mid < best_r:
            slope_r, best_r = slope, x2 - x_mid
    # Get the average lines of the two clusters
    lx1, lx2, rx1, rx2 = [], [], [], []
    for x1, y1, x2, y2 in lines:
        if abs((float(y2) - y1) / (x2 - x1) - slope_l) < 0.2:
            mc = np.polyfit([x1, x2], [y1, y2], 1)
            lx1.append(np.int(np.float((y_min - mc[1])) / np.float(mc[0])))
            lx2.append(np.int(np.float((y_max - mc[1])) / np.float(mc[0])))
        elif abs((float(y2) - y1) / (x2 - x1) - slope_r) < 0.2:
            mc = np.polyfit([x1, x2], [y1, y2], 1)
            rx1.append(np.int(np.float((y_min - mc[1])) / np.float(mc[0])))
            rx2.append(np.int(np.float((y_max - mc[1])) / np.float(mc[0])))
    if len(lx1) == 0 or len(lx2) == 0 or len(rx1) == 0 or len(rx2) == 0:
        return image
    lx1_avg = np.int(np.nanmean(lx1))
    lx2_avg = np.int(np.nanmean(lx2))
    rx1_avg = np.int(np.nanmean(rx1))
    rx2_avg = np.int(np.nanmean(rx2))
    midx1_avg = (lx1_avg + rx1_avg) / 2
    midx2_avg = (lx2_avg + rx2_avg) / 2
    left = [(lx1_avg, y_min), (lx2_avg, y_max)]
    right = [(rx1_avg, y_min), (rx2_avg, y_max)]
    middle = [(midx1_avg, y_min), (midx2_avg, y_max)]
    cv2.line(image, (left[0][0], left[0][1]), (left[1][0], left[1][1]), (255, 0, 255), 8)
    cv2.line(image, (right[0][0], right[0][1]), (right[1][0], right[1][1]), (255, 0, 255), 8)
    cv2.line(image, (middle[0][0], middle[0][1]), (middle[1][0], middle[1][1]), (0, 255, 0), 4)
    return image
    # return [(lx1_avg, y_min), (lx2_avg, y_max)], \
    #        [(rx1_avg, y_min), (rx2_avg, y_max)], \
    #        [(midx1_avg, y_min), (midx2_avg, y_max)]
    # except:
    #     return -1,-1,-1

