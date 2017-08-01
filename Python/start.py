from __future__ import division
import cv2
import numpy as np
import time
from helpers import *

def still_image():
    straight = cv2.imread('images/track5.jpg', 0)
    slant = cv2.imread('images/track_slant_resize.png', 0)

    image = LinedImage(straight)
    image.set_canny(200, 255)
    image.set_hough(1, np.pi / 180, 100)
    start = time.time()
    image.run_avg(min_slope=50)
    # image.get_lines()
    print(time.time() - start)

    image.display(only_lines=True)
    image.save('track_slant')

def video_lanes():
    cap = cv2.VideoCapture('videos/straight_short.mp4')
    ret, frame = cap.read()
    fourcc = cv2.cv.CV_FOURCC(*'SVQ3')
    out = cv2.VideoWriter('output_py.avi', fourcc, 10.0, (frame.shape[1], frame.shape[0]))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            lined_frame = frame.copy()
            lined_frame = track_lanes(lined_frame)
            out.write(lined_frame)
            cv2.imshow('lines', lined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    stabilize('videos/straight.mp4')
