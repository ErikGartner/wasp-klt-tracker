import cv2


class Tracker(object):

    def __init__(self):
        self.current_image = None
        self.current_roi = None

    def new_roi(self, img, roi):
        """
        Sets the region of interest.
        """
        self.current_image = img
        self.current_roi = roi
        self._update_features()

    def _update_features(self):
        """
        Selectes features to track in the ROI.
        """
        pass

    def next_image(img_arr):
        """
        Takes the next frame and tracks the ROI using the selected
        features. Returns a new ROI.
        """
        pass
