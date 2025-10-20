"""
Disparity tuner with fine-grained scaling, cropping, filters, and profiles.
Press:
  q – quit and save current settings
  s – save profile
  l – load profile
Profiles are saved in ./disparity_profiles/<name>.json
"""

from __future__ import annotations
from turtle import right
import cv2, json, os, numpy as np
import cv2.ximgproc as xip
from src.hal.cam.calibrate.calib import load_calibration
from src.hal.cam.Camera import open_stereo_pair

import threading, time

ROOT = os.path.join(os.path.dirname(__file__), "../../..")
SETTINGS_FILE = os.path.join(ROOT, "disparity_settings.json")
PROFILE_DIR = os.path.join(ROOT, "disparity_profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)

class PerfTracker:
    def __init__(self):
        self.times = {}
        self.last = time.perf_counter()

    def mark(self, label):
        now = time.perf_counter()
        self.times[label] = (now - self.last) * 1000  # ms
        self.last = now

    def summary(self):
        total = sum(self.times.values())
        parts = " | ".join(f"{k}:{v:.1f}ms" for k,v in self.times.items())
        return f"{parts} | total:{total:.1f}ms"


DEFAULT_SETTINGS = {
    "minDisparity": 0,
    "numDisparities": 4,
    "blockSize": 16,
    "preFilterCap": 31,
    "uniquenessRatio": 5,
    "speckleWindowSize": 160,
    "speckleRange": 7,
    "disp12MaxDiff": 7,
    "medianBlurK": 0,
    "downSample": 100,
    "crop": 0,
    "farEnhance": 50,
    "nearCutoff": 0,
    # filters
    "useMorph": 1,
    "morphIter": 1,
    "useBilateral": 1,
    "bilateralStrength": 8,
    "useWLS": 0,
    "wlsLambda": 4000,
    "wlsSigma": 1.0,
    "profileName": "default"
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    return DEFAULT_SETTINGS.copy()

def save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)

def save_profile(s):
    path = os.path.join(PROFILE_DIR, f"{s['profileName']}.json")
    with open(path, "w") as f:
        json.dump(s, f, indent=2)
    print(f"✅ Saved profile: {path}")

def load_profile(name):
    path = os.path.join(PROFILE_DIR, f"{name}.json")
    if not os.path.exists(path):
        print(f"⚠️ Profile '{name}' not found.")
        return None
    with open(path) as f:
        return {**DEFAULT_SETTINGS, **json.load(f)}

def nothing(x): pass

def create_tuner_window(s):
    cv2.namedWindow("Disparity Tuner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Disparity Tuner", 480, 600)
    # Core SGBM
    cv2.createTrackbar("minDisp", "Disparity Tuner", s["minDisparity"], 100, nothing)
    cv2.createTrackbar("numDisp", "Disparity Tuner", s["numDisparities"], 20, nothing)
    cv2.createTrackbar("blockSize", "Disparity Tuner", s["blockSize"], 21, nothing)
    cv2.createTrackbar("uniqueness", "Disparity Tuner", s["uniquenessRatio"], 50, nothing)
    cv2.createTrackbar("preFilterCap", "Disparity Tuner", s["preFilterCap"], 63, nothing)
    cv2.createTrackbar("speckleRange", "Disparity Tuner", s["speckleRange"], 50, nothing)
    cv2.createTrackbar("medianBlurK", "Disparity Tuner", s["medianBlurK"], 7, nothing)
    cv2.createTrackbar("downSample%", "Disparity Tuner", s["downSample"], 100, nothing)
    cv2.createTrackbar("crop(px)", "Disparity Tuner", s["crop"], 200, nothing)
    # Filters
    cv2.createTrackbar("useMorph", "Disparity Tuner", s["useMorph"], 1, nothing)
    cv2.createTrackbar("morphIter", "Disparity Tuner", s["morphIter"], 3, nothing)
    cv2.createTrackbar("useBilateral", "Disparity Tuner", s["useBilateral"], 1, nothing)
    cv2.createTrackbar("bilateralStrength", "Disparity Tuner", s["bilateralStrength"], 20, nothing)
    cv2.createTrackbar("useWLS", "Disparity Tuner", s["useWLS"], 1, nothing)
    cv2.createTrackbar("wlsLambda", "Disparity Tuner", s["wlsLambda"], 10000, nothing)
    cv2.createTrackbar("wlsSigmaX10", "Disparity Tuner", int(s["wlsSigma"] * 10), 50, nothing)
    cv2.createTrackbar("farEnhance", "Disparity Tuner", s.get("farEnhance", 50), 200, nothing)
    cv2.createTrackbar("nearCutoff", "Disparity Tuner", s.get("nearCutoff", 0), 200, nothing)



def read_trackbar():
    s = {
        "minDisparity": cv2.getTrackbarPos("minDisp", "Disparity Tuner"),
        "numDisparities": max(1, cv2.getTrackbarPos("numDisp", "Disparity Tuner")),
        "blockSize": max(3, cv2.getTrackbarPos("blockSize", "Disparity Tuner") | 1),
        "preFilterCap": cv2.getTrackbarPos("preFilterCap", "Disparity Tuner"),
        "uniquenessRatio": cv2.getTrackbarPos("uniqueness", "Disparity Tuner"),
        "speckleRange": cv2.getTrackbarPos("speckleRange", "Disparity Tuner"),
        "speckleWindowSize": 160,
        "disp12MaxDiff": 7,
        "medianBlurK": cv2.getTrackbarPos("medianBlurK", "Disparity Tuner"),
        "downSample": max(10, cv2.getTrackbarPos("downSample%", "Disparity Tuner")),
        "crop": cv2.getTrackbarPos("crop(px)", "Disparity Tuner"),
        "useMorph": cv2.getTrackbarPos("useMorph", "Disparity Tuner"),
        "morphIter": cv2.getTrackbarPos("morphIter", "Disparity Tuner"),
        "useBilateral": cv2.getTrackbarPos("useBilateral", "Disparity Tuner"),
        "bilateralStrength": cv2.getTrackbarPos("bilateralStrength", "Disparity Tuner"),
        "useWLS": cv2.getTrackbarPos("useWLS", "Disparity Tuner"),
        "wlsLambda": cv2.getTrackbarPos("wlsLambda", "Disparity Tuner"),
        "farEnhance": cv2.getTrackbarPos("farEnhance", "Disparity Tuner"),
        "nearCutoff": cv2.getTrackbarPos("nearCutoff", "Disparity Tuner"),

        "wlsSigma": cv2.getTrackbarPos("wlsSigmaX10", "Disparity Tuner") / 10.0
    }
    if s["medianBlurK"] % 2 == 0:
        s["medianBlurK"] = max(0, s["medianBlurK"] - 1)
    return s

def preprocess_images(grayL, grayR, s):
    scale = max(0.1, s["downSample"] / 100.0)
    if scale < 0.999:
        grayL = cv2.resize(grayL, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        grayR = cv2.resize(grayR, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    c = s["crop"]
    if c > 0:
        h, w = grayL.shape[:2]
        grayL = grayL[c:h - c, c:w - c]
        grayR = grayR[c:h - c, c:w - c]
    return grayL, grayR


def compute_disparity_map(gray_left, gray_right, settings):
    min_disp = settings["minDisparity"]
    num_disp = 16 * settings["numDisparities"]
    blk = settings["blockSize"]
    P1, P2 = 8 * blk * blk, 32 * blk * blk
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=blk,
        P1=P1, P2=P2,
        preFilterCap=settings["preFilterCap"],
        uniquenessRatio=settings["uniquenessRatio"],
        speckleWindowSize=settings["speckleWindowSize"],
        speckleRange=settings["speckleRange"],
        disp12MaxDiff=settings["disp12MaxDiff"],
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    if settings["medianBlurK"] >= 3:
        disp = cv2.medianBlur(disp, settings["medianBlurK"])
    return disp, num_disp

def post_filter_strong(disp, grayL, s):
    if s["useMorph"]:
        disp = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=s["morphIter"])
    if s["useBilateral"] and s["bilateralStrength"] > 0:
        b = s["bilateralStrength"]
        disp = cv2.bilateralFilter(disp, 5, b, b)
    if s["useWLS"] and s["wlsLambda"] > 0:
        wls = xip.createDisparityWLSFilterGeneric(False)
        wls.setLambda(float(s["wlsLambda"]))
        wls.setSigmaColor(float(s["wlsSigma"]))
        disp = wls.filter(disp, grayL)
    return disp

def post_filter_weak(disp, s):
    """Post-filter disparity: remove very near/high-disparity pixels and
    perform iterative neighborhood cleanup to remove weakly-connected remnants.

    Args:
        disp: disparity map (numpy array, float)
        s: settings dict (expects key 'nearCutoff')

    Returns:
        Filtered disparity map (same shape/type as input)
    """
    # remove close objects (high disparity, high pass filter)
    if s.get("nearCutoff", 0) > 0:
        cutoff = s["nearCutoff"]
        disp[disp > cutoff] = 0

    # --- Iterative neighborhood cleanup ---
    mask = (disp > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    # repeat a few rounds to remove weakly connected remnants
    for _ in range(3):
        neighbor_count = cv2.filter2D(mask, -1, kernel)
        # remove pixels with too few valid neighbors (≤3 of 8)
        isolated = (neighbor_count <= 3) & (mask == 1)
        if not np.any(isolated):
            break
        mask[isolated] = 0

    disp[mask == 0] = 0
    # --- end cleanup ---

    return disp

def visualize_disparity(disp, num_disp, far_enhance=50):
    # clamp to valid range
    disp = np.clip(disp, 0, num_disp)

    # normalize window shift: 0–200 slider → bias toward far field
    # far_enhance=0 → normal contrast
    # far_enhance=200 → focus only on far (low disparity)
    bias = np.clip(far_enhance / 200.0, 0.0, 1.0)

    # compute percentile window
    valid = disp[disp > 0]
    if valid.size > 0:
        low = np.percentile(valid, (1 - bias) * 80)
        high = np.percentile(valid, 100 - (1 - bias) * 10)
        if high <= low:
            high = low + 1
        disp_vis = np.clip((disp - low) / (high - low), 0, 1)
    else:
        disp_vis = np.zeros_like(disp)

    # apply colormap
    norm = (disp_vis * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return color

def rectify_pair(left, right, calib):
    leftMapX, leftMapY, rightMapX, rightMapY, _, _ = calib
    rectL = cv2.remap(left, leftMapX, leftMapY, cv2.INTER_LINEAR)
    rectR = cv2.remap(right, rightMapX, rightMapY, cv2.INTER_LINEAR)
    return rectL, rectR


def remove_void_rows(disp, threshold=0.95, min_neighbors=5):
    """
    Remove rows that are mostly void (zeros) or sparsely populated with valid disparity.
    Args:
        disp: disparity map (float32)
        threshold: fraction of zeros above which a row is removed (0–1)
        min_neighbors: minimum valid pixels per row to keep it
    Returns:
        disparity map with removed or zeroed rows
    """
    mask = disp > 0
    h, w = mask.shape
    for y in range(h):
        valid_count = np.count_nonzero(mask[y])
        if valid_count < min_neighbors or valid_count / w < (1 - threshold):
            disp[y, :] = 0
    return disp

class ThreadedCamera:
    def __init__(self, cam):
        self.cam = cam
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        t = threading.Thread(target=self._update, daemon=True)
        t.start()

    def _update(self):
        while self.running:
            f = self.cam.read_frame()
            if f is not None:
                with self.lock:
                    self.frame = f
            else:
                time.sleep(0.005)

    def get(self):
        with self.lock:
            return self.frame

    def close(self):
        self.running = False
        self.cam.close()


def main():
    calib = load_calibration()
    left_cam_raw, right_cam_raw = open_stereo_pair()
    left_cam = ThreadedCamera(left_cam_raw)
    right_cam = ThreadedCamera(right_cam_raw)
    s = load_settings()
    create_tuner_window(s)
    print("Press 's' to save current profile, 'l' to load one, 'q' to quit.")

    try:
        while True:
            tracker = PerfTracker()
            left = left_cam.get()
            right = right_cam.get()
            tracker.mark("capture")

            if left is None or right is None:
                time.sleep(0.001)
                continue

            rectL, rectR = rectify_pair(left, right, calib)
            tracker.mark("rectify")
            grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

            s = read_trackbar()
            grayL, grayR = preprocess_images(grayL, grayR, s)
            tracker.mark("preproc")

            disp, num_disp = compute_disparity_map(grayL, grayR, s)
            tracker.mark("sgbm")

            # postfilter-strong: bilateral, WLS, etc.
            disp = post_filter_strong(disp, grayL, s)

            # postfilter-weak: remove very-near pixels and cleanup small remnants
            disp = post_filter_weak(disp, s)

            disp = remove_void_rows(disp)
            tracker.mark("filter")


            # visualize
            vis = visualize_disparity(disp, num_disp, s["farEnhance"])
            tracker.mark("vis")

            vis_display = cv2.resize(vis, (1920, 1080))



            cv2.putText(vis, f"Profile={s.get('profileName','default')} | DS={s['downSample']}% | Crop={s['crop']}px",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(vis, f"Filters: M{'✔' if s['useMorph'] else '✖'}  B{'✔' if s['useBilateral'] else '✖'}  W{'✔' if s['useWLS'] else '✖'}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.imshow("Disparity (color)", vis_display)
            print(tracker.summary())

            key = cv2.waitKey(1) & 0xFF

            #  Save/load profiles, and quit
            if key == ord("s"):
                name = input("Enter profile name to save: ").strip()
                if name:
                    s["profileName"] = name
                    save_profile(s)
            elif key == ord("l"):
                name = input("Enter profile name to load: ").strip()
                prof = load_profile(name)
                if prof:
                    s = prof
                    save_settings(s)
                    create_tuner_window(s)
            elif key == ord("q"):
                save_settings(s)
                break

    # Handle cleanup on exit
    finally:
        left_cam.close()
        right_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
