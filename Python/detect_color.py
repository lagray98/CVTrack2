import numpy as np
import argparse
import cv2


def detectColor(image):

    boundaries = [
#                  ([17, 15, 100], [110, 110, 255]) #red mask (BGR instead of RGB)
#                  ([160, 150, 200], [255, 255, 255]) #white mask
                  ([0,0,0],[115,115,115])
                  ]

    for (lower, upper) in boundaries:

        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        mask = cv2.inRange(image, lower, upper)
       
        
        output = cv2.bitwise_and(image, image, mask = mask)
      

#        cv2.imshow("images", np.hstack([image, output]))
#        cv2.waitKey(0)
        return output #<type 'numpy.ndarray'>
