"""
Detector
========
One job: take a frame, return the detections that belong to TARGET classes.

The interface it exposes is just two things:
    det = Detector()          # build once
    detections = det(frame)   # call per frame -> list of Detection

How it runs YOLO, which device it uses, etc. is hidden from the outside
(encapsulation). If you swap the model later, this interface stays the same.
"""
from ultralytics import YOLO

from ..types import Detection
from .. import config


class Detector:
    def __init__(self, model_path=None, device=None,
                 target_names=None, min_conf=None):
        # Defaults come from config; can be overridden by caller
        self.model = YOLO(model_path or config.MODEL_PATH)
        self.device = device or config.DEVICE
        self.target_names = target_names or config.TARGET_NAMES
        self.min_conf = min_conf if min_conf is not None else config.MIN_CONF

    def __call__(self, frame):
        """
        frame: BGR NumPy array (H, W, 3)
        returns: list of Detection for target classes (may be empty)
        """
        results = self.model(frame, device=self.device, verbose=False)
        boxes = results[0].boxes
        detections = []

        if len(boxes) == 0:
            return detections

        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        for i in range(len(confs)):
            name = self.model.names[clss[i]]
            if name in self.target_names and confs[i] >= self.min_conf:
                x1, y1, x2, y2 = xyxy[i]
                detections.append(Detection(
                    cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    cls_name=name, conf=float(confs[i]),
                ))
        return detections
