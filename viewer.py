import sys
import json
import argparse

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


class Viewer:

    def __init__(self, image_stream, tracker):

        self.tracker = tracker
        self.image_stream = image_stream
        self.current_image = next(self.image_stream)

        # collects unfinished bounding box
        self.bounding_box = []

        # Create figure
        self.fig, self.ax = plt.subplots()

    def launch_viewer(self, block=True):
        """
        Creates a plot and gives over execution control to matplotlib.
        """
        cid1 = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        cid2 = self.fig.canvas.mpl_connect('button_press_event',
                                           self._on_click)
        self.update_view()
        plt.show(block=block)

    def update_view(self):
        """
        Redraws the current node with annotations
        """
        self.ax.clear()
        im = self.current_image
        self.ax.imshow(im)
        #self.ax.axis('off')
        self.fig.canvas.draw()

    def _show_bounding_box(self, a, color='red', label=None):
        """
        Draws bounding boxes to image.
        """
        pos = (a[0][0], a[0][1])
        self.ax.add_patch(
            patches.Rectangle(
                pos,
                a[1][0] - a[0][0],
                a[1][1] - a[0][1],
                fill=False,
                color=color
            ))
        self.fig.canvas.draw()

    def _show_features(self, features):
        for f in features:
            pos = (int(f[0, 0]), int(f[0, 1]))
            p = patches.Circle(pos,
                               radius=3, color='red')
            self.ax.add_patch(p)
        self.fig.canvas.draw()

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

    def _next_frame(self):
        """
        Tracks the roi to the next frame.
        """
        # Get next frame
        self.current_image = next(self.image_stream)
        # Show next frame
        self.update_view()
        roi = self.tracker.next_image(self.current_image)

        # Show the new box we're tracking
        self._show_bounding_box(roi)

    def _on_click(self, event):
        """
        Handles the on click event.
        """

        if event.inaxes is None:
            # outside of images
            return

        if event.xdata is None or event.ydata is None:
            # Invalid click pos
            return

        if len(self.bounding_box) < 4:
            # Collect clicks to the bounding box
            self.bounding_box.append((event.xdata, event.ydata))

        if len(self.bounding_box) == 4:
            box = self._get_bbox()
            self._show_bounding_box(box)

            self._start_tracker(box)

    def _on_key(self, event):
        self._next_frame()
