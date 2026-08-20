"""
Test detector + tracker + camera compensation TOGETHER (single target)
======================================================================
This is the modular equivalent of the old track_video.py, but built from
independent modules. It picks the single most confident detection, feeds its
camera-compensated center to one Tracker, and draws the lock + AIM point.

Run:
    python -m antiair.test_tracker

Keys:  space=pause  .=step(when paused)  c=toggle compensation  q=quit
"""
import cv2

from .perception.detector import Detector
from .perception.tracker import Tracker
from .perception.camera_motion import CameraMotion
from . import config


def main():
    det = Detector()
    trk = Tracker(track_id=1)
    cam = CameraMotion()

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: could not open '{config.VIDEO_PATH}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
                    continue
                break

            h, w = frame.shape[:2]

            # 1) detect
            dets = det(frame)

            # 2) camera motion (exclude target boxes from the background estimate)
            boxes = [d.box for d in dets]
            cam.update(frame, boxes)
            cdx, cdy = cam.total

            # 3) pick most confident detection, compensate its center, feed tracker
            measurement = None
            conf = None
            if dets:
                best = max(dets, key=lambda d: d.conf)
                measurement = (best.cx - cdx, best.cy - cdy)
                conf = best.conf
                cv2.circle(frame, (int(best.cx), int(best.cy)), 6, (0, 0, 255), -1)
                cv2.putText(frame, f"{best.cls_name} {best.conf:.2f}",
                            (int(best.x1), int(best.y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                            (0, 0, 255), config.FONT_THICK)

            track = trk.update(measurement, conf)

            # 4) draw lock + leading (add camera shift back to draw on screen)
            if track is not None:
                sx = track.cx + cdx
                sy = track.cy + cdy
                px, py = int(sx), int(sy)
                color = (0, 255, 0) if track.locked else (0, 165, 255)
                cv2.circle(frame, (px, py), 16, color, 3)
                cv2.line(frame, (px - 26, py), (px + 26, py), color, 2)
                cv2.line(frame, (px, py - 26), (px, py + 26), color, 2)

                nx = int(px + track.vx * lead_frames)
                ny = int(py + track.vy * lead_frames)
                cv2.line(frame, (px, py), (nx, ny), (255, 0, 255), 2)
                cv2.circle(frame, (nx, ny), 13, (255, 0, 255), 3)
                cv2.putText(frame, "AIM", (nx + 16, ny),
                            cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                            (255, 0, 255), config.FONT_THICK)
            else:
                cv2.putText(frame, "NO TARGET", (12, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                            (0, 0, 255), config.FONT_THICK)

            comp = "COMP: ON" if config.COMPENSATE else "COMP: OFF"
            cv2.putText(frame, comp, (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        config.FONT_SCALE, (0, 255, 255), config.FONT_THICK)

            cv2.imshow("tracker test", frame)

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
