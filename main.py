import imageio
import cv2
import numpy as np

from viewer import Viewer
from tracker import Tracker, KLTTracker


if __name__ == '__main__':

    # Get test video
    reader = imageio.get_reader('/Users/erik/Desktop/coke_zero.mp4')
    reader = iter(reader)

    tracker = Tracker()

    viewer = Viewer(reader, tracker, cv2.COLOR_RGB2BGR)
    viewer.launch_viewer()
