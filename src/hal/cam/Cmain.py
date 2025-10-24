import os
import cv2
from datetime import datetime

from src.hal.cam.Camera import open_stereo_pair
from src.hal.cam.calibrate.calib import load_calibration
from src.hal.cam.Depth import DisparityDepthCapture

PREVIEW = False
SAVE = False
OUT_DIR = "./images"


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def main() -> None:
    calib = load_calibration()
    engine = DisparityDepthCapture(calibration=calib, default_profile="CDR")

    left, right = open_stereo_pair()

    try:
        while True:
            frameL = left.read_frame()
            frameR = right.read_frame()
            if frameL is None or frameR is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            res = engine.process(frameL, frameR)

            if PREVIEW:
                vis = engine.visualize(res["disp"], res["num_disp"])
                cv2.imshow("Depth", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if SAVE:
                os.makedirs(OUT_DIR, exist_ok=True)
                path = os.path.join(OUT_DIR, f"depth_{ts()}.npz")
                engine.save_npz(path, res["depth"], res["num_disp"], res["meta"])

    finally:
        left.close()
        right.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
