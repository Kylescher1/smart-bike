import os
import cv2
from datetime import datetime
# Stereo depth capture main loop for Smart Bike HAL camera.
# - Grabs frames from a calibrated stereo pair
# - Runs disparity/depth via DisparityDepthCapture
# - Optionally previews a colorized disparity and/or saves depth as NPZ

from src.hal.cam.Camera import open_stereo_pair
from src.hal.cam.calibrate.calib import load_calibration
from src.hal.cam.Depth import DisparityDepthCapture

# Toggle live visualization of disparity (non-blocking UI). Press 'q' to quit.
PREVIEW = False
# Toggle saving depth outputs to OUT_DIR as compressed NPZ files.
SAVE = False
OUT_DIR = "./images"

# Timestamp helper for filenames (ms precision)
def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def main() -> None:
    # Load stereo rectification + Q matrix from calibration files.
    calib = load_calibration()
    # Processing engine: handles rectification, disparity, filters, depth.
    engine = DisparityDepthCapture(calibration=calib, default_profile="CDR")

    # Open left/right camera handles (no compute here; just I/O wrappers).
    left, right = open_stereo_pair()

    try:
        # Main acquisition/processing loop
        while True:
            # Read the latest frames (BGR) from each camera.
            frameL = left.read_frame()
            frameR = right.read_frame()
            # If a frame is missing, keep the UI responsive and retry.
            if frameL is None or frameR is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Core computation: returns {'depth','disp','num_disp','meta'}
            res = engine.process(frameL, frameR)

            # Optional on-screen preview of disparity for quick checks.
            if PREVIEW:
                vis = engine.visualize(res["disp"], res["num_disp"])
                cv2.imshow("Depth", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Optional save of calibrated depth results to disk.
            if SAVE:
                os.makedirs(OUT_DIR, exist_ok=True)
                path = os.path.join(OUT_DIR, f"depth_{ts()}.npz")
                engine.save_npz(path, res["depth"], res["num_disp"], res["meta"])

    # Always release cameras and close any OpenCV windows.
    finally:
        left.close()
        right.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
