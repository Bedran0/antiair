"""
MultiTracker (with motion-based re-ID)
======================================
Follows several targets at once AND tries to keep identities stable across
short disappearances (target leaves frame / detection drops for a while).

Two-stage matching each frame:
  A) ACTIVE match (IoU): new detections continue existing active trackers,
     greedy highest-IoU first.
  B) RE-ID match (motion): detections that matched no active tracker are
     compared against "limbo" -- recently lost targets. Each limbo entry's
     position is extrapolated along its last known velocity to "now"; if a
     detection lands within REID_RADIUS of that predicted spot, the old
     IDENTITY is revived. We use MOTION only, never appearance, because in
     this scenario the targets look identical (same plane model/paint).

  Leftover detections -> brand-new identities.
  Active trackers with no match -> coast; after MAX_LOST they move to limbo.
  Limbo entries older than LIMBO_SECONDS are forgotten for good.

Known limit (as the user pointed out): two identical targets that vanish and
return on overlapping trajectories can still be swapped -- motion can't
separate them, and appearance won't help when they look the same. A single
camera fundamentally can't resolve that; real systems add radar / IFF.
"""
from dataclasses import dataclass

from .tracker import Tracker
from ..utils import iou
from .. import config


@dataclass
class LimboEntry:
    """A recently lost target, remembered briefly for re-identification."""
    id: int
    cx: float          # last known position (compensated coords)
    cy: float
    vx: float          # last known velocity
    vy: float
    cls_name: str
    box_w: float
    box_h: float
    age: int = 0       # frames spent in limbo


class MultiTracker:
    def __init__(self, fps=30.0):
        self.trackers = []
        self.limbo = []
        self._next_id = 1
        self._fps = fps if fps and fps > 0 else 30.0

    def set_fps(self, fps):
        self._fps = fps if fps and fps > 0 else 30.0

    def _new_tracker(self, det, cam_dx, cam_dy, reuse_id=None):
        """Create a tracker, feed it the detection once, return (tracker, Track)."""
        tid = reuse_id if reuse_id is not None else self._next_id
        if reuse_id is None:
            self._next_id += 1
        t = Tracker(track_id=tid, cls_name=det.cls_name)
        meas = (det.cx - cam_dx, det.cy - cam_dy)
        size = (det.x2 - det.x1, det.y2 - det.y1)
        trk = t.update(meas, conf=det.conf, box_size=size)
        self.trackers.append(t)
        return t, trk

    def update(self, detections, cam_dx=0.0, cam_dy=0.0):
        # ---------- A) ACTIVE match via IoU (+ center-distance fallback) ----------
        # A detection continues a track if their boxes overlap enough (IoU) OR
        # their centers are close enough. The center fallback survives sudden
        # box-size jumps (e.g. smoke/occlusion splitting the box) that would
        # otherwise drop IoU below threshold and break the identity.
        pairs = []
        for di, d in enumerate(detections):
            for ti, t in enumerate(self.trackers):
                pbox = t.predicted_box(cam_dx, cam_dy)
                score = iou(d.box, pbox)
                pcx = (pbox[0] + pbox[2]) / 2
                pcy = (pbox[1] + pbox[3]) / 2
                dist = ((d.cx - pcx) ** 2 + (d.cy - pcy) ** 2) ** 0.5
                if score >= config.IOU_MATCH_THRESHOLD or dist <= config.CENTER_MATCH_DIST:
                    combined = score + max(0.0, 1.0 - dist / config.CENTER_MATCH_DIST)
                    pairs.append((combined, di, ti))
        pairs.sort(reverse=True)

        det_taken, trk_taken, matches = set(), set(), {}
        for score, di, ti in pairs:
            if di in det_taken or ti in trk_taken:
                continue
            matches[ti] = di
            det_taken.add(di)
            trk_taken.add(ti)

        tracks = []

        # update matched active trackers, coast the rest
        survivors = []
        for ti, t in enumerate(self.trackers):
            if ti in matches:
                d = detections[matches[ti]]
                meas = (d.cx - cam_dx, d.cy - cam_dy)
                size = (d.x2 - d.x1, d.y2 - d.y1)
                trk = t.update(meas, conf=d.conf, box_size=size)
            else:
                trk = t.update(None)                # coast
            if t.alive:
                survivors.append(t)
                if trk is not None:
                    tracks.append(trk)
            else:
                # died -> push to limbo with its last known state
                cx, cy = t.predicted_center()
                self.limbo.append(LimboEntry(
                    id=t.id, cx=cx, cy=cy,
                    vx=float(t.kf.statePost[2][0]), vy=float(t.kf.statePost[3][0]),
                    cls_name=t.cls_name, box_w=t.box_w, box_h=t.box_h))
        self.trackers = survivors

        # ---------- B) RE-ID unmatched detections against limbo ----------
        for di, d in enumerate(detections):
            if di in det_taken:
                continue

            reuse_id = self._match_limbo(d, cam_dx, cam_dy)
            _, tr = self._new_tracker(d, cam_dx, cam_dy, reuse_id=reuse_id)
            det_taken.add(di)
            if tr is not None:
                tracks.append(tr)

        # ---------- age & prune limbo ----------
        max_age = int(config.LIMBO_SECONDS * self._fps)
        for e in self.limbo:
            e.age += 1
            # extrapolate its predicted position along last velocity
            e.cx += e.vx
            e.cy += e.vy
        self.limbo = [e for e in self.limbo if e.age <= max_age]

        return tracks

    def _match_limbo(self, det, cam_dx, cam_dy):
        """
        Return an ID to revive if this detection lands near a limbo target's
        extrapolated position, else None (brand-new target).
        """
        mx, my = det.cx - cam_dx, det.cy - cam_dy      # compensated
        best_id, best_dist = None, config.REID_RADIUS
        best_index = -1
        for idx, e in enumerate(self.limbo):
            dist = ((mx - e.cx) ** 2 + (my - e.cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = e.id
                best_index = idx
        if best_index >= 0:
            self.limbo.pop(best_index)                  # consume it
        return best_id

    def reset(self):
        self.trackers = []
        self.limbo = []
        self._next_id = 1
