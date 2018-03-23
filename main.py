import imageio

from viewer import Viewer
from tracker import Tracker


if __name__ == '__main__':

    # Get test video
    reader = imageio.get_reader('/Users/erik/Desktop/coke_zero.mp4')
    reader = iter(reader)

    tracker = Tracker()

    viewer = Viewer(reader, tracker)
    viewer.launch_viewer()
