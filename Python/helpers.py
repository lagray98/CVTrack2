import cv2
import numpy as np
import math

BLUE_COLOR = (255, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)


def stabilize(file_name, output_name):
    """
    Stabilizes input video, displays in window, and writes to output

    @param file_name: input video file name
    @type file_name: str
    @param output_name: output video file name
    @type output_name: str
    """

    cap = cv2.VideoCapture(file_name)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=200,
                          qualityLevel=0.01,
                          minDistance=20,
                          blockSize=10)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    rows, cols, _ = old_frame.shape

    output_video = cv2.VideoWriter(output_name, cv2.cv.CV_FOURCC(*'SVQ3'), 30.0, (cols, rows))

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    ret, frame = cap.read()
    shift_x = 0
    shift_y = 0
    while ret:
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
        shift_x += np.mean(x)
        shift_y += np.mean(y)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        output_frame = cv2.warpAffine(frame, M, dsize=(cols, rows))
        output_video.write(output_frame)
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
    output_video.release()


def canny_mask(img):
    """
    Applies Canny Edge detection and masks relevant area of an image

    @param img: image to alter
    @type img: Any
    @return: altered image
    @rtype: Any
    """
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


class LineCluster:
    """
    Cluster of similar lines
    """

    def __init__(self, y_min, y_max, (x1, y1), (x2, y2), allowed_error=5000):
        """
        Initialize new LineCluster
        @param y_min: Minimum y value for lines
        @type y_min: int
        @param y_max: Maximum y value for lines
        @type y_max: int
        @param allowed_error: Maximum squared distance between lines in the same cluster
        @type allowed_error: int or float
        """
        # NOTE: slope_range percent difference
        self.bottom_line, self.top_line = [], []
        self.y_min = y_min
        self.y_max = y_max
        self.error = allowed_error
        slope, intercept = np.polyfit([x1, x2], [y1, y2], 1)
        self.bottom_line.append(np.int(np.float((self.y_min - intercept)) / np.float(slope)))
        self.top_line.append(np.int(np.float((self.y_max - intercept)) / np.float(slope)))

    def check_append(self, point1, point2):
        """
        Check if new line is part of the cluster, then appends it if applicable
        @param point1: First point of the new line
        @type point1: (int, int)
        @param point2: Second point of the new line
        @type point2: (int, int)
        @return: Boolean indicating if the new line has been included
        @rtype: bool
        """
        x1, y1 = point1
        x2, y2 = point2
        slope, intercept = np.polyfit([x1, x2], [y1, y2], 1)
        bottom_x = np.int(np.float((self.y_min - intercept)) / np.float(slope))
        top_x = np.int(np.float((self.y_max - intercept)) / np.float(slope))
        mse = (np.square(np.abs(np.mean(self.bottom_line) - bottom_x))    #Should be square root, not divided by 2 ?
               + np.square(np.abs(np.mean(self.top_line) - top_x)))/2
        if mse < self.error:
            self.bottom_line.append(bottom_x)
            self.top_line.append(top_x)
            return True
        else:
            return False

    def get_line(self):
        """
        Returns the mean line of the cluster
        @return: Tuple of two points indicating the mean line of the cluster
        @rtype: ((int, int), (int, int))
        """
        return (np.int(np.nanmean(self.bottom_line)), self.y_min), (np.int(np.nanmean(self.top_line)), self.y_max)

    def bottom_x(self):
        """
        Returns the x-coordinate of the bottom point of the mean line of the cluster
        @return: X-coordinate of the bottom point of the mean line of the cluster
        @rtype: int
        """
        return np.int(np.mean(self.bottom_line))

    def top_x(self):
        """
        Returns the x-coordinate of the top point of the mean line of the cluster
        @return: X-coordinate of the top point of the mean line of the cluster
        @rtype: int
        """
        return np.int(np.mean(self.top_line))


def trace_lines(image, lines, color=BLUE_COLOR):
    """
    Traces lines over input image
    @param image: Image to trace lines over
    @type image: Any
    @param lines: List of lines
    @type lines: list of ((int, int), (int, int)) or list of (int, int, int, int)
    @param color: Tuple indicating color of lines to be traced in form (r, g, b)
    @type color: (int, int, int)
    @return: Traced over image
    @rtype: Any
    """
    for line in lines:
        if len(line) == 2:
            x1, y1, x2, y2 = line[0][0], line[0][1], line[1][0], line[1][1]
        elif len(line) == 4:
            x1, y1, x2, y2 = line
        else:
            return
        if abs(y2 - y1) < 50:
            continue
        cv2.line(image, (x1, y1), (x2, y2), color, 5)
    return image


def average_line_variable_cluster(lines, bounds):
    """
    Clusters lines into a variable number of clusters
    @param lines: Lines to cluster
    @type lines: list of (int, int, int, int)
    @param bounds: Min and Max values for output lines (min_x, max_x, min_y, max_y)
    @type bounds: (int, int, int, int)
    @return: Clusters of lines
    @rtype: list of LineCluster
    """
    clusters = []
    for x1, y1, x2, y2 in lines:
        if y2-y1 == 0:
            continue
        if x2-x1 == 0:
            x2 -= 1

        check = False
        for cluster in clusters:
            # Difference in slope < 0.2 and difference in intercept < 10% of cluster's intercept
            if cluster.check_append((x1, y1), (x2, y2)):
                check = True
                break
        if not check:
            new_cluster = LineCluster(bounds[1], bounds[3], (x1, y1), (x2, y2))
            clusters.append(new_cluster)

    return clusters


def track_lanes(image):
    """
    Finds left and right side of lane, middle of lane, and POV orientation in lane of image
    @param image: image to process
    @type image: Any
    @return: Image with all lines traced over
    @rtype: Any
    """
    y_min = image.shape[0]
    y_max = int(image.shape[0] * 0.4)
    x_min = 0
    x_max = image.shape[1]
    canny = canny_mask(image)
    lines = cv2.HoughLinesP(canny, 1, np.pi/200, 4,
                            minLineLength=y_min*0.2, maxLineGap=y_min*0.05)
    if lines is None:
        return image
    lines = lines[0]
    clustered_lines = average_line_variable_cluster(lines, (x_min, y_min, x_max, y_max))

    x_mid = np.int(np.float(x_max) / 2)
    r_line, rx = None, float("Inf")
    l_line, lx = None, float("-Inf")
    for line in clustered_lines:
        x_dist = (x_mid - line.bottom_x()) + (x_mid - line.top_x())
        if rx > x_dist > 0:
            r_line, rx = line, x_dist
        elif 0 > x_dist > lx:
            l_line, lx = line, x_dist
    r_line = r_line.get_line()
    l_line = l_line.get_line()
    base_mid = (r_line[0][0] + l_line[0][0])/2.0
    base_width = r_line[0][0] - l_line[0][0]
    deviation = (x_mid - base_mid) / base_width

    if deviation < 0:
        print "Too Far Right: ", deviation,"%"
    elif deviation > 0:
        print "Too Far Left: ", deviation, "%"
    else:
        print "Correct"
    mid_line = [((r_line[0][0] + l_line[0][0])/2, (r_line[0][1] + l_line[0][1])/2),((r_line[1][0]+l_line[1][0])/2, (r_line[1][1] + l_line[1][1])/2)]
    user_line = [(x_mid, y_min), mid_line[1]]
    image = trace_lines(image, [mid_line], color=GREEN_COLOR)
    image = trace_lines(image, [user_line], color=BLUE_COLOR)
    return trace_lines(image, [r_line, l_line], color=RED_COLOR)


def findAngle(leftLine, rightLine, minY):
    # [bottom x, top x]
    leftXDistance = leftLine[1] - leftLine[0]
    leftLineLength = np.sqrt(np.square(minY)+ np.square(leftXDistance))
    leftAngle = math.asin(minY/leftLineLength) * 180 / math.pi
#    print("Left Angle: " + str(leftAngle))

    rightXDistance = rightLine[0] - rightLine[1]
    rightLineLength = np.sqrt(np.square(minY) + np.square(rightXDistance))
    rightAngle = math.asin(minY/rightLineLength)*180 / math.pi
#    print("Right Angle: " + str(rightAngle))

    topAngle = 180 - leftAngle - rightAngle
#    print("Top Angle: " + str(topAngle))
    return topAngle
