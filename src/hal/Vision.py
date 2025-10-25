import os
import cv2
from datetime import datetime
# Stereo depth capture main loop for Smart Bike HAL camera.
# - Grabs frames from a calibrated stereo pair
# - Runs disparity/depth via DisparityDepthCapture
# - Optionally previews a colorized disparity and/or saves depth as NPZ

from src.hal.cam.Camera import open_stereo_pair
from src.hal.cam.calibrate.calib import load_calibration
from src.hal.cam.depth_processor import DisparityDepthCapture

# Toggle live visualization of disparity (non-blocking UI). Press 'q' to quit.
PREVIEW = False
# Toggle saving depth outputs to OUT_DIR as compressed NPZ files.
SAVE = False
OUT_DIR = "./images"

# Timestamp helper for filenames (ms precision)
def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def visualize_disparity(disp, num_disp, *, colormap: str = "jet", far_enhance: int = 50):
    """
    Colorized visualization tuned towards far field when far_enhance > 0.
    Returns a BGR uint8 image.
    """
    import numpy as np

    d = np.clip(disp, 0, num_disp).astype(np.float32)
    valid = d[d > 0]
    if valid.size > 0:
        import numpy as np  # local to keep Vision lightweight on import
        bias = max(0.0, min(1.0, far_enhance / 200.0))
        low = float(np.percentile(valid, (1 - bias) * 80))
        high = float(np.percentile(valid, 100 - (1 - bias) * 10))
        if high <= low:
            high = low + 1.0
        vis = np.clip((d - low) / (high - low), 0.0, 1.0)
    else:
        vis = np.zeros_like(d)

    norm = (vis * 255.0).astype(np.uint8)
    if colormap == "bw":
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    elif colormap == "bone":
        return cv2.applyColorMap(norm, cv2.COLORMAP_BONE)
    else:
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

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
                far = engine.get_settings().get("farEnhance", 50)
                vis = visualize_disparity(res["disp"], res["num_disp"], colormap="jet", far_enhance=far)
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
