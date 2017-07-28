from __future__ import division
import cv2
from helpers import LinedImage
import numpy as np
import time
straight = cv2.imread('images/track5.jpg', 0)
slant = cv2.imread('images/track_slant_resize.png', 0)

image = LinedImage(straight)
image.set_canny(200, 255)
image.set_hough(1, np.pi/180, 100)
start = time.time()
image.run(min_slope=50)
# image.get_lines()
print(time.time() - start)


image.display(only_lines=True)
image.save('track_slant')

# for line in linedImage.lines:


