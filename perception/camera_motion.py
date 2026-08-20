"""
CameraMotion
============
Estimates how much the CAMERA moved between two frames, using sparse optical
flow on the background. This is scene-level information (one value per frame,
independent of any target), so it lives in its own module. In multi-target
mode we compute it once and apply it to every track.

Interface:
    cam = CameraMotion()
    dx, dy = cam.update(frame, target_boxes)   # per-frame cumulative shift is tracked internally
    cam.total                                  # (cum_dx, cum_dy) since start

The key idea: screen_motion = target_real_motion + camera_motion.
By measuring camera_motion from the background, we can subtract it and
recover the target's real motion -- so aiming stays correct even when the
camera pans to follow the target.
"""
import numpy as np
import cv2

from .. import config


class CameraMotion:
    def __init__(self):
        self.prev_gray = None
        self.cum_dx = 0.0            # cumulative camera shift since start
        self.cum_dy = 0.0
        self.last_dx = 0.0           # last successful per-frame shift (for coasting)
        self.last_dy = 0.0

    def reset(self):
        self.prev_gray = None
        self.cum_dx = self.cum_dy = 0.0
        self.last_dx = self.last_dy = 0.0

    @property
    def total(self):
        return (self.cum_dx, self.cum_dy)

    def _frame_shift(self, gray, target_boxes):
        """Median background shift between prev_gray and gray. (0,0) if it fails."""
        pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None,
                                      **config.FEATURE_PARAMS)
        if pts is None:
            return None

        # Drop points that fall inside any target box -- those move with the
        # target, not the camera.
        if target_boxes:
            keep = []
            for p in pts:
                px, py = p.ravel()
                inside = any(x1 <= px <= x2 and y1 <= py <= y2
                             for (x1, y1, x2, y2) in target_boxes)
                if not inside:
                    keep.append(p)
            pts = np.array(keep, dtype=np.float32) if keep else None

        if pts is None or len(pts) < config.MIN_FLOW_POINTS:
            return None

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, pts, None, **config.LK_PARAMS)
        good_old = pts[status.ravel() == 1]
        good_new = nxt[status.ravel() == 1]
        if len(good_new) < config.MIN_FLOW_POINTS:
            return None

        # Median is robust: a few stray target points can't skew it.
        flow = (good_new - good_old).reshape(-1, 2)
        return float(np.median(flow[:, 0])), float(np.median(flow[:, 1]))

    def update(self, frame, target_boxes):
        """
        Call once per frame. Updates the cumulative camera shift and returns
        this frame's (dx, dy).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or not config.COMPENSATE:
            self.prev_gray = gray
            return 0.0, 0.0

        shift = self._frame_shift(gray, target_boxes)
        if shift is None:
            # Flow failed this frame -> assume camera kept moving the same way
            # (coast), just like the Kalman filter coasts when it loses a measurement.
            dx, dy = self.last_dx, self.last_dy
        else:
            dx, dy = shift
            self.last_dx, self.last_dy = dx, dy

        self.cum_dx += dx
        self.cum_dy += dy
        self.prev_gray = gray
        return dx, dy
