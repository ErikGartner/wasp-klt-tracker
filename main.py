import imageio
import cv2
import numpy as np

from viewer import Viewer
from tracker import Tracker, KLTTracker


class StreamIterator():

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)

    def __iter__(self):
        return self

    def next(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            raise StopIteration()

    def __next__(self):
        return self.next()


if __name__ == '__main__':

    source = 0 # For webcam
    source = 'data/coke_zero.mp4'

    # Get test video
    stream = StreamIterator(source)
    for i in range(2):
        # Skip first 2 frames to "warm up" the webcam
        next(stream)

    tracker = Tracker()

    viewer = Viewer(stream, tracker)
    viewer.launch_viewer()

    cap.release()
    cv2.destroyAllWindows()
