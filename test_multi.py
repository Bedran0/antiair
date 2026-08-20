"""
Test MULTI-target tracking
==========================
detector -> camera compensation -> MultiTracker. Every target gets its own
color and a persistent ID drawn on screen, plus its own AIM point.

Run:
    python -m antiair.test_multi

Keys:  space=pause  c=toggle compensation  q=quit
"""
import cv2

from .perception.detector import Detector
from .perception.multi_tracker import MultiTracker
from .perception.camera_motion import CameraMotion
from . import config

# a few distinct colors, cycled by track id
COLORS = [(0, 255, 0), (255, 128, 0), (0, 200, 255),
          (255, 0, 255), (0, 255, 255), (255, 255, 0)]


def color_for(track_id):
    return COLORS[track_id % len(COLORS)]


def main():
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: could not open '{config.VIDEO_PATH}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    det = Detector()
    mtrk = MultiTracker(fps=fps)
    cam = CameraMotion()
    lead_frames = config.FLIGHT_TIME * fps
    print(f"Video FPS: {fps:.1f} -> leading {lead_frames:.0f} frames")

    paused = False
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                if config.LOOP_VIDEO:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    cam.reset()
                    mtrk.reset()
                    continue
                break

            h, w = frame.shape[:2]

            dets = det(frame)
            cam.update(frame, [d.box for d in dets])
            cdx, cdy = cam.total

            tracks = mtrk.update(dets, cdx, cdy)

            for tr in tracks:
                col = color_for(tr.id)
                # position on screen = compensated + camera shift
                px = int(tr.cx + cdx)
                py = int(tr.cy + cdy)
                # dim the color while coasting
                c = col if tr.locked else tuple(int(v * 0.5) for v in col)

                cv2.circle(frame, (px, py), 14, c, 3)
                cv2.putText(frame, f"ID {tr.id} {tr.cls_name}",
                            (px + 18, py - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, c, 2)

                nx = int(px + tr.vx * lead_frames)
                ny = int(py + tr.vy * lead_frames)
                cv2.line(frame, (px, py), (nx, ny), (255, 0, 255), 2)
                cv2.circle(frame, (nx, ny), 8, (255, 0, 255), 2)

            cv2.putText(frame, f"Targets: {len(tracks)}", (12, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, (0, 255, 255), 2)
            comp = "COMP: ON" if config.COMPENSATE else "COMP: OFF"
            cv2.putText(frame, comp, (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        config.FONT_SCALE, (0, 255, 255), 2)

            cv2.imshow("multi-target test", frame)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("c"):
            config.COMPENSATE = not config.COMPENSATE
            cam.reset()
            print(f"Compensation: {'ON' if config.COMPENSATE else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
