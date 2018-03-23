import cv2
import numpy as np


class Tracker(object):

    def __init__(self):
        self.current_image = None
        self.roi = None

    def _process_image(self, img):
        return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    def new_roi(self, img, roi):
        """
        Sets the region of interest.
        """
        self.current_image = self._process_image(img)
        self.roi = roi
        self._update_features()

    def _update_features(self):
        """
        Selectes features to track in the ROI.
        """
        # Create binary mask
        mask = np.zeros_like(self.current_image)
        mask[self.roi[0][1]:self.roi[1][1], self.roi[0][0]:self.roi[1][0]] = 1

        # Get features from opencv
        corners = cv2.goodFeaturesToTrack(self.current_image, maxCorners=10,
                                          qualityLevel=0.05, minDistance=5,
                                          mask=mask)
        self.features = corners

    def get_features(self):
        return self.features

    def next_image(self, img):
        """
        Takes the next frame and tracks the ROI using the selected
        features. Returns a new ROI.
        """
        return self.roi
