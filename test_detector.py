"""
Test the Detector IN ISOLATION
===============================
No Kalman, no leading, no compensation -- just "does the detector find the
target in a frame?" Being able to test modules in isolation is the biggest
payoff of a modular architecture.

Run (from the PARENT of antiair/, as a package):
    python -m antiair.test_detector
"""
import cv2

from .perception.detector import Detector
from . import config


def main():
    det = Detector()
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: could not open '{config.VIDEO_PATH}'.")
        return

    print("Detector test started. Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            if config.LOOP_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        dets = det(frame)   # <-- all the magic in one line

        for d in dets:
            cv2.rectangle(frame, (int(d.x1), int(d.y1)),
                          (int(d.x2), int(d.y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"{d.cls_name} {d.conf:.2f}",
                        (int(d.x1), int(d.y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                        (0, 0, 255), config.FONT_THICK)

        cv2.putText(frame, f"Detections: {len(dets)}", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, (0, 255, 255), 2)
        cv2.imshow("detector test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
