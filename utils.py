"""
Geometry helpers
================
Small, pure functions shared across modules. Keeping them here avoids
duplicating the same math in several places.
"""


def iou(box_a, box_b):
    """
    Intersection over Union of two boxes, each (x1, y1, x2, y2).
    Returns 0.0 (no overlap) .. 1.0 (identical). Used for data association:
    a new detection that overlaps an existing track a lot is probably the
    same object.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection rectangle
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
