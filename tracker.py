import cv2
import numpy as np


class Tracker(object):

    def __init__(self):
        self.current_image = None
        self.roi = None
        self.feature_history = []

    def _process_image(self, img):
        return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    def new_roi(self, img, roi):
        """
        Sets the region of interest.
        """
        self.current_image = self._process_image(img)
        self.roi = roi
        self._update_features()
        self.feature_history = []

    def _update_features(self):
        """
        Selectes features to track in the ROI.
        """
        # Create binary mask
        mask = np.zeros_like(self.current_image)
        mask[self.roi[0][1]:self.roi[1][1], self.roi[0][0]:self.roi[1][0]] = 1

        # Get features from opencv
        corners = cv2.goodFeaturesToTrack(self.current_image, maxCorners=10,
                                          qualityLevel=0.05, minDistance=3,
                                          mask=mask)
        self.features = corners

    def get_features(self):
        return self.features

    def next_image(self, img, calc_new_features=True):
        """
        Takes the next frame and tracks the ROI using the selected
        features. Returns a new ROI.
        """
        # Process the image to the correct format.
        img = self._process_image(img)

        # Track the features
        new_pos = self._track_features(img)

        # Update the roi based on the tracked features
        self.roi = self._update_roi(self.features, new_pos)

        # Update to the next image
        self.current_image = img
        self.feature_history.append(self.features)

        if calc_new_features:
            self._update_features()
        else:
            self.features = new_pos
        return self.roi

    def _track_features(self, new_img):
        """
        Tracks the feature points.
        """
        pts = self.features.astype(np.float32)
        I2 = self.current_image
        J2 = new_img
        res = cv2.calcOpticalFlowPyrLK(I2, J2, pts, None, winSize=(21, 21), maxLevel=0)
        return res[0]

    def _update_roi(self, old_features, new_features):
        """
        Update the roi according to the movement of the features.
        """
        diff = new_features - old_features
        x_diff = np.round(np.mean(diff[:, :, 0]))
        y_diff = np.mean(diff[:, :, 1])
        x_start = int(np.round(self.roi[0][0] + x_diff))
        y_start = int(np.round(self.roi[0][1] + y_diff))
        x_end = int(np.round(self.roi[1][0] + x_diff))
        y_end = int(np.round(self.roi[1][1] + y_diff))
        roi = [(x_start, y_start),
               (x_end, y_end)]
        return roi
