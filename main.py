import argparse

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


def parse_args():
    parser = argparse.ArgumentParser(description='KTL Tracker')
    parser.add_argument('source', nargs='?', help='The source, defaults to webcame, else filepath to a video.', default=0)
    parser.add_argument('--custom', action='store_true', help='use my custom KTL implementation.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    # Get test video
    stream = StreamIterator(args.source)
    for i in range(2):
        # Skip first 2 frames to "warm up" the webcam
        next(stream)

    if args.custom:
        tracker = KLTTracker()
    else:
        tracker = Tracker()

    viewer = Viewer(stream, tracker)
    viewer.launch_viewer()

    cap.release()
    cv2.destroyAllWindows()
