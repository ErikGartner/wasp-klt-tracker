import sys
import json
import argparse

import numpy as np
import cv2


class Viewer:

    def __init__(self, image_stream, tracker):

        self.tracker = tracker
        self.image_stream = image_stream

        self.current_image = next(self.image_stream)

        # collects unfinished bounding box
        self.bounding_box = []

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self._on_click)

    def launch_viewer(self):
        self.update_view()
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                return

            elif key == 13:
                self._next_frame()

    def update_view(self):
        """
        Redraws the current node with annotations
        """
        cv2.imshow('image', self.current_image)

    def _show_bounding_box(self, a, color='red', label=None):
        """
        Draws bounding boxes to image.
        """
        cv2.rectangle(self.current_image, a[0], a[1], (0, 255, 0), 3)

    def _show_features(self, features):
        for f in features:
            pos = (int(f[0, 0]), int(f[0, 1]))
            cv2.circle(self.current_image, pos, 3, (0, 0, 255), -1)

    def _get_bbox(self):
        """
        Adds a bounding box from the points in bounding_box.
        """
        x_min = int(round(min([p[0] for p in self.bounding_box])))
        x_max = int(round(max([p[0] for p in self.bounding_box])))
        y_min = int(round(min([p[1] for p in self.bounding_box])))
        y_max = int(round(max([p[1] for p in self.bounding_box])))

        width = self.current_image.shape[1]
        height = self.current_image.shape[0]

        box = [(max(x_min, 0), max(y_min, 0)),
               (min(x_max, width), min(y_max, height))]

        self.bounding_box = []
        return box

    def _start_tracker(self, box):
        """
        Starts the tracker for the given ROI.
        """
        self.tracker.new_roi(self.current_image, box)
        features = self.tracker.get_features()
        self._show_features(features)
        self.update_view()

    def _next_frame(self):
        """
        Tracks the roi to the next frame.
        """
        # Get next frame
        self.current_image = next(self.image_stream)

        roi = self.tracker.next_image(self.current_image)

        # Show the new box we're tracking
        self._show_bounding_box(roi)

        # Show the features
        features = self.tracker.get_features()
        self._show_features(features)

        # Show next frame
        self.update_view()

    def _on_click(self, event, x, y, flags, param):
        """
        Handles the on click event.
        """

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(self.bounding_box) < 4:
            # Collect clicks to the bounding box
            self.bounding_box.append((x, y))

        if len(self.bounding_box) == 4:
            box = self._get_bbox()
            self._show_bounding_box(box)
            self._start_tracker(box)

    def _on_key(self, event):
        self._next_frame()
