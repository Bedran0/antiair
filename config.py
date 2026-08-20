"""
Central configuration
=====================
All constants live here. To change something, look here instead of hunting
through the code.
"""

# --- Input ---
VIDEO_PATH = "/Users/bedran/antiair/test_video.mp4"     # video file (use an absolute path when running
                                  # as a module, e.g. /Users/you/antiair/test.mp4);
                                  # pass 0/1 in a script for a webcam
LOOP_VIDEO = True

# --- Detector ---
# For the fine-tuned aerial model use its weights + class names, e.g.
#   MODEL_PATH = "/Users/you/antiair/runs/detect/train-2/weights/best.pt"
#   TARGET_NAMES = {"drone", "helicopter", "plane", "birds"}
# For the stock COCO model use "yolo26n.pt"/"yolo26s.pt" with COCO names.
MODEL_PATH = "/Users/bedran/yolo26n.pt"
DEVICE = "mps"                    # "mps" (Apple), "cuda" (NVIDIA), "cpu"
TARGET_NAMES = {"airplane", "bird", "kite"}
MIN_CONF = 0.35

# --- Tracker (Kalman) ---
MAX_LOST = 30                     # frames a tracker coasts before going to limbo
MEAS_NOISE = 1.0                  # measurementNoiseCov (chosen by sweep; 1.0 good)
PROCESS_NOISE = 0.03

# --- Data association / re-ID ---
IOU_MATCH_THRESHOLD = 0.2         # min IoU for a detection to continue a track
CENTER_MATCH_DIST = 80            # px: centers this close also count as a match
                                  # (survives box-size jumps from occlusion)
# When a tracker dies it waits in "limbo" for a while. If a new detection
# appears near where the lost target WOULD be (last position + velocity *
# elapsed), it revives with the OLD id. Motion-based only -- no appearance,
# which is why two identical planes on the same path can still be confused.
LIMBO_SECONDS = 7.0               # how long a lost id is remembered for re-ID
REID_RADIUS = 250                 # px: how close a new detection must be to revive

# --- Camera-motion compensation (optical flow) ---
COMPENSATE = True
FEATURE_PARAMS = dict(maxCorners=500, qualityLevel=0.001,
                      minDistance=8, blockSize=7)
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3)
MIN_FLOW_POINTS = 3

# --- Leading / firing ---
FLIGHT_TIME = 0.5                 # projectile time to target (seconds)
HIT_RADIUS = 30

# --- Visualization ---
FONT_SCALE = 2.0
FONT_THICK = 2
MIRROR = False                    # may be needed for webcam; False for video
