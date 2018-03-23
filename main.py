import imageio
import cv2

from viewer import Viewer
from tracker import Tracker, KLTTracker


if __name__ == '__main__':

    # Get test video
    reader = imageio.get_reader('/Users/erik/Desktop/coke_zero.mp4')
    reader = iter(reader)

    tracker = KLTTracker()

    viewer = Viewer(reader, tracker)
    viewer.launch_viewer()
