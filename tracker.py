import cv2


class Tracker(object):

    def __init__(self):
        self.current_image = None
        self.current_roi = None

    def new_roi(self):
        """
        Sets the region of interest.
        """
        self._select_roi()
        self._update_features()

    def _select_roi(self):
        coordinates = input('x1,y1,x2,y2').split(',')
        self.current_roi = [(coordinates[0], coordinates[1]),
                            (coordinates[2], coordinates[3])]

    def _update_features(self):
        """
        Selectes features to track in the ROI.
        """
        pass

    def _next_image(img_arr):
        """
        Takes the next frame and tracks the ROI using the selected
        features.
        """
        pass

    def show_image():
        """
        Shows the current state of the tracking.
        """
        pass
