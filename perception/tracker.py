"""
Tracker (single target)
========================
Wraps one constant-velocity Kalman filter. Given a measurement it maintains a
smooth, identified Track (position + velocity) and coasts (predicts) when no
measurement arrives.

For multi-target association it also remembers the last box SIZE, so it can
report a predicted box (predicted center + last known size). MultiTracker
uses that box to match new detections via IoU.

This class knows nothing about YOLO, cameras, or drawing. In multi-target
mode we create one Tracker per target and reuse it unchanged.
"""
import numpy as np
import cv2

from ..types import Track
from .. import config


class Tracker:
    def __init__(self, track_id=0, cls_name="?"):
        self.id = track_id
        self.cls_name = cls_name
        self.initialized = False
        self.lost = 0
        self.conf = 0.0
        self.box_w = 0.0            # last known box width  (for predicted box)
        self.box_h = 0.0            # last known box height

        kf = cv2.KalmanFilter(4, 2)          # state: [x, y, vx, vy]
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * config.PROCESS_NOISE
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * config.MEAS_NOISE
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.zeros((4, 1), np.float32)
        self.kf = kf

    def predicted_center(self):
        """Where the filter thinks the target is right now (compensated coords)."""
        return float(self.kf.statePost[0][0]), float(self.kf.statePost[1][0])

    def predicted_box(self, cam_dx=0.0, cam_dy=0.0):
        """
        Predicted box in SCREEN coords (center + last known size).
        cam_dx/cam_dy: add camera shift back, since the filter runs in
        compensated space but detections come in screen space.
        """
        cx, cy = self.predicted_center()
        cx += cam_dx
        cy += cam_dy
        hw, hh = self.box_w / 2, self.box_h / 2
        return (cx - hw, cy - hh, cx + hw, cy + hh)

    def update(self, measurement, conf=None, box_size=None):
        """
        measurement: (x, y) in COMPENSATED coords, or None if not seen.
        box_size:    (w, h) of the matched detection's box (screen size), or None.
        returns:     current Track, or None if never initialized.
        """
        if measurement is not None:
            mx, my = measurement
            meas = np.array([[np.float32(mx)], [np.float32(my)]])
            if not self.initialized:
                self.kf.statePost = np.array([[mx], [my], [0], [0]], np.float32)
                self.initialized = True
            self.kf.predict()
            est = self.kf.correct(meas)
            self.lost = 0
            if conf is not None:
                self.conf = conf
            if box_size is not None:
                self.box_w, self.box_h = box_size
        elif self.initialized:
            est = self.kf.predict()          # coast
            self.lost += 1
            if self.lost > config.MAX_LOST:
                self.initialized = False
        else:
            return None

        return Track(
            id=self.id,
            cx=float(est[0][0]), cy=float(est[1][0]),
            vx=float(est[2][0]), vy=float(est[3][0]),
            cls_name=self.cls_name, conf=self.conf,
            lost=self.lost, has_lock=(measurement is not None),
        )

    @property
    def alive(self):
        return self.initialized
