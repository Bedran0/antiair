# antiair

Real-time detection and tracking of aerial objects (planes, helicopters, drones, birds) in a video or camera stream. Built from small, independent, testable modules: YOLO detection, single- and multi-target Kalman tracking, optical-flow camera-motion compensation, and motion-based re-identification. Includes aim-prediction (leading) for a downstream turret, but the core is a general-purpose perception pipeline usable in any aircraft-detection system.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-0B23A9?logo=ultralytics&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Compute-013243?logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Backend-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Demo

Two aircraft tracked at once with stable IDs, camera-motion compensation, and leading (the magenta line predicts where to aim):

![demo](docs/example.gif)

## What it does

The pipeline turns a stream of frames into stable, identified tracks. Per frame:

1. Detect aerial targets with a YOLO model (filtered by class and confidence).
2. Estimate camera motion from background optical flow and subtract it, so a panning camera does not corrupt target velocity.
3. Associate detections to existing tracks (IoU plus a center-distance fallback), assigning each a persistent ID.
4. Smooth each target with a constant-velocity Kalman filter; coast (predict) through short detection gaps.
5. Re-identify targets that briefly leave the frame using motion-based extrapolation.
6. Predict a lead point (where the target will be after the projectile's flight time).

## Architecture

```
antiair/
  config.py                 all settings in one place
  types.py                  Detection, Track dataclasses (shared vocabulary)
  utils.py                  iou()
  perception/
    detector.py             YOLO detection, class + confidence filtering
    camera_motion.py        optical-flow camera-motion compensation
    tracker.py              single-target constant-velocity Kalman filter
    multi_tracker.py        multi-target association + limbo re-identification
  coordination/             (reserved) turret-to-target assignment
  turret/                   (reserved) turret abstraction + driver interface
  viz/                      (reserved) drawing helpers
  test_detector.py          run the detector alone
  test_tracker.py           detector + tracker + compensation (single target)
  test_multi.py             multi-target tracking with IDs
```

Each module has one job and a narrow interface, so any part (for example the detector) can be swapped without touching the rest.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/antiair.git
cd antiair
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install ultralytics opencv-python numpy
```

A YOLO model file is downloaded automatically on first run when you reference a stock model name (for example `yolo26n.pt`).

## Configuration

Edit `config.py`:

- `VIDEO_PATH` — absolute path to your video (relative paths break when running as a module). Use `0` for a webcam in a script.
- `MODEL_PATH` — YOLO weights. Stock COCO (`yolo26n.pt`, `yolo26s.pt`) recognizes `airplane`, `bird`, `kite` but not drones or helicopters. A model fine-tuned on aerial data can recognize `drone`, `helicopter`, `plane`, `birds`.
- `TARGET_NAMES` — the class names to keep, matching your model.
- `DEVICE` — `mps` (Apple Silicon), `cuda` (NVIDIA), or `cpu`.

## Usage

Run from the parent directory of `antiair/`, as a package:

```bash
cd ..
python -m antiair.test_detector    # detection only
python -m antiair.test_tracker     # single-target track + leading + compensation
python -m antiair.test_multi       # multi-target tracking with IDs
```

Keys in the test windows: `space` pause, `c` toggle camera compensation, `q` quit.

### Using the pipeline in your own code

```python
from antiair.perception.detector import Detector
from antiair.perception.multi_tracker import MultiTracker
from antiair.perception.camera_motion import CameraMotion

detector = Detector()
tracker = MultiTracker(fps=30)
camera = CameraMotion()

for frame in your_frames:
    detections = detector(frame)
    camera.update(frame, [d.box for d in detections])
    cdx, cdy = camera.total
    tracks = tracker.update(detections, cdx, cdy)
    # each track has: id, cx, cy, vx, vy, cls_name, conf, locked
```

## Design decisions

These were chosen by measurement, not assumption:

- **Constant-velocity Kalman**, not an acceleration model. Acceleration is far too noise-sensitive; the `t^2` term in the leading equation amplifies the error. Verified in a synthetic test harness where the acceleration model was consistently worse.
- **Camera-motion compensation** subtracts background optical-flow shift so a camera panning to follow a target does not appear as target motion. It coasts (repeats the last shift) when flow fails on textureless sky.
- **Motion-based re-identification only**, no appearance. Targets are often visually identical (same aircraft model and paint), so appearance cannot separate them. Known limitation: two identical targets that vanish and return on overlapping paths can be swapped. A single camera cannot resolve this; real systems add radar or IFF.

## Notes and limitations

- The stock COCO model cannot detect drones or helicopters (not among its 80 classes); those require fine-tuning on a custom aerial dataset.
- Optical-flow compensation needs background texture; on perfectly clear sky it has few features to track and falls back to coasting.
- Detection quality depends on the model. A model fine-tuned with imbalanced class counts will be strong on well-represented classes and weak on rare ones.

## License

MIT
