"""
Shared data types
==================
Every module speaks these types. The detector produces a Detection, the
tracker turns it into a Track. This shared "language" keeps modules loosely
coupled -- changing one won't break the others.

dataclass: Python's lightweight data container that auto-generates __init__,
__repr__, etc. Saves writing boilerplate constructors.
"""
from dataclasses import dataclass


@dataclass
class Detection:
    """A single object found in one frame (raw YOLO output)."""
    cx: float          # box center x
    cy: float          # box center y
    x1: float          # box top-left x
    y1: float          # box top-left y
    x2: float          # box bottom-right x
    y2: float          # box bottom-right y
    cls_name: str      # class name ("airplane", "bird"...)
    conf: float        # confidence score 0..1

    @property
    def box(self):
        """(x1, y1, x2, y2) tuple -- e.g. for optical-flow masking."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Track:
    """
    A target followed over time. Unlike a Detection, it has an IDENTITY (id)
    that persists across frames. Position/velocity come from the Kalman filter.
    """
    id: int                       # unique identity
    cx: float                     # (compensated) position x
    cy: float                     # position y
    vx: float = 0.0               # velocity x (px/frame)
    vy: float = 0.0               # velocity y
    cls_name: str = "?"           # class name
    conf: float = 0.0             # last confidence
    lost: int = 0                 # frames since last measurement
    has_lock: bool = False        # got a real measurement this frame?

    @property
    def locked(self):
        return self.has_lock and self.lost == 0
