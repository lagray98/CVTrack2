from __future__ import division
import time
from helpers import *
from LaneTracker import LaneTracker


# def video_lanes():
#     cap = cv2.VideoCapture('videos/overcorrect_left.mp4')
#     ret, frame = cap.read()
#     fourcc = cv2.cv.CV_FOURCC(*'SVQ3')
#     out = cv2.VideoWriter('overcorrect_left_output.avi', fourcc, 30.0, (frame.shape[1], frame.shape[0]))
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             lined_frame = frame.copy()
#             lined_frame = track_lanes(lined_frame)
#             out.write(lined_frame)
#             cv2.imshow('lines', lined_frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             break
#
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#

def stabilizer():
    stabilize('videos/straight_walk.mp4','output_video.avi')


if __name__ == '__main__':
    """
    Overlays predicted track in input video and outputs to file and view window
    """
    input_file = 'videos/overcorrect_left.mp4' # Can be mp4 or avi
    output_file = 'overcorrect_left_out.avi'

    cap = cv2.VideoCapture(input_file)
    ret, frame = cap.read()
    out = cv2.VideoWriter(output_file, cv2.cv.CV_FOURCC(*'SVQ3'), 30.0, (frame.shape[1], frame.shape[0]))
    tracker = LaneTracker(frame.shape[0], frame.shape[0] * 0.4)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            tracker.step(frame)
            polygon = tracker.polygon()
            polygon = polygon.reshape((-1, 1, 2))
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], GREEN_COLOR)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            out.write(frame)
            cv2.imshow('track', frame)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

    cv2.destroyAllWindows()
    cap.release()
    out.release()