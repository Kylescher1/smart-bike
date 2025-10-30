"""Entry point for the Smart Bike vision pipeline."""
from __future__ import annotations

import time

import cv2

from src.hal.Vision import VisionSystem, default_calibration_file


def main() -> None:
    calibration_path = default_calibration_file()
    vision = VisionSystem(calibration_file=calibration_path)

    vision.open()
    print("✅ Vision system initialised. Press Ctrl+C to stop.")

    # -----------------------------
    # Live disparity tuner (no save)
    # -----------------------------
    def ensure_odd(n: int, min_val: int = 3) -> int:
        n = max(min_val, n)
        return n if (n % 2 == 1) else (n + 1)

    def create_tuner_window() -> None:
        cv2.namedWindow("Disparity Tuner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Disparity Tuner", 420, 640)
        p = vision._profile_params or {}
        def tb(name, maxv, init):
            cv2.createTrackbar(name, "Disparity Tuner", int(init), int(maxv), lambda v: None)

        tb("numDisparitiesK", 12, int(p.get("numDisparitiesK", 4)))
        tb("blockSize", 21, int(p.get("blockSize", 5)))
        tb("minDisparity", 32, int(p.get("minDisparity", 0)))
        tb("preFilterCap", 63, int(p.get("preFilterCap", 31)))
        tb("uniquenessRatio", 40, int(p.get("uniquenessRatio", 10)))
        tb("speckleWindowSize", 256, int(p.get("speckleWindowSize", 100)))
        tb("speckleRange", 64, int(p.get("speckleRange", 32)))
        tb("disp12MaxDiff", 32, int(p.get("disp12MaxDiff", 1)))
        tb("downSample", 100, int(p.get("downSample", 100)))
        tb("crop", 200, int(p.get("crop", 0)))
        tb("farEnhance", 200, int(p.get("farEnhance", 50)))
        tb("nearCutoff", 255, int(p.get("nearCutoff", 0)))
        tb("morphIter", 5, int(p.get("morphIter", 1)))
        tb("bilateralStrength", 20, int(p.get("bilateralStrength", 8)))
        tb("wlsLambda", 5000, int(p.get("wlsLambda", 0)))
        tb("wlsSigma_x10", 100, int(round(float(p.get("wlsSigma", 1.0)) * 10)))
        # binary toggles as 0/1
        tb("useMorph", 1, int(p.get("useMorph", 1)))
        tb("useBilateral", 1, int(p.get("useBilateral", 1)))
        tb("useWLS", 1, int(p.get("useWLS", 0)))

    def read_tuner_params() -> dict:
        g = cv2.getTrackbarPos
        params = dict(vision._profile_params or {})
        params["numDisparitiesK"] = max(1, int(g("numDisparitiesK", "Disparity Tuner")))
        params["blockSize"] = ensure_odd(int(g("blockSize", "Disparity Tuner")), 3)
        params["minDisparity"] = int(g("minDisparity", "Disparity Tuner"))
        params["preFilterCap"] = int(g("preFilterCap", "Disparity Tuner"))
        params["uniquenessRatio"] = int(g("uniquenessRatio", "Disparity Tuner"))
        params["speckleWindowSize"] = int(g("speckleWindowSize", "Disparity Tuner"))
        params["speckleRange"] = int(g("speckleRange", "Disparity Tuner"))
        params["disp12MaxDiff"] = int(g("disp12MaxDiff", "Disparity Tuner"))
        params["downSample"] = max(10, int(g("downSample", "Disparity Tuner")))
        params["crop"] = max(0, int(g("crop", "Disparity Tuner")))
        params["farEnhance"] = int(g("farEnhance", "Disparity Tuner"))
        params["nearCutoff"] = int(g("nearCutoff", "Disparity Tuner"))
        params["morphIter"] = int(g("morphIter", "Disparity Tuner"))
        params["bilateralStrength"] = int(g("bilateralStrength", "Disparity Tuner"))
        params["wlsLambda"] = int(g("wlsLambda", "Disparity Tuner"))
        params["wlsSigma"] = float(g("wlsSigma_x10", "Disparity Tuner")) / 10.0
        params["useMorph"] = int(g("useMorph", "Disparity Tuner"))
        params["useBilateral"] = int(g("useBilateral", "Disparity Tuner"))
        params["useWLS"] = int(g("useWLS", "Disparity Tuner"))
        return params

    create_tuner_window()

    # Ensure the Depth map window is 1920x1080
    cv2.namedWindow("Depth map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth map", 1920, 1080)

    try:
        while True:
            frames = vision.capture_frames()
            if frames is None:
                continue

            left_frame, right_frame = frames
            # Update live params from tuner
            vision._profile_params = read_tuner_params()
            depth_result = vision.compute_depth(left_frame, right_frame)
            edge_map = VisionSystem.edge_map_from_depth(depth_result.depth_map)
            depth_vis = depth_result.depth_map
            depth_color = None
            if depth_vis.size:
                norm = cv2.normalize(depth_vis.astype("float32"), None, 0, 255, cv2.NORM_MINMAX)
                norm = 255 - norm  # invert so nearer (smaller depth) becomes larger -> warmer after JET
                depth_color = cv2.applyColorMap(norm.astype("uint8"), cv2.COLORMAP_JET)

            shown = False
            if depth_color is not None:
                cv2.imshow("Depth map", depth_color)
                shown = True
            if edge_map.size:
                cv2.imshow("Depth edges", edge_map)
                shown = True
            if shown:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if vision.is_object_close(depth_result.depth_map):
                vision.warn_rider()

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("⏹️  Vision loop interrupted by user.")
    finally:
        vision.close()
        cv2.destroyAllWindows()
        print("👋 Vision system shut down.")
        # --- ASK USER TO SAVE SETTINGS TO DEFAULT PROFILE ---
        try:
            ans = input("Do you want to save the current disparity settings to the default profile? (y/n): ").strip().lower()
            if ans == 'y':
                from src.hal.cam.depth_profile import save_settings, PROFILE_DIR
                from src.hal.config import PROFILE_NAME
                import os, json
                params = vision._profile_params or {}
                # Save to disparity_settings.json (legacy/global)
                save_settings(params)
                # Save to default profile (by name)
                profile_path = os.path.join(PROFILE_DIR, f"{PROFILE_NAME}.json")
                os.makedirs(PROFILE_DIR, exist_ok=True)
                with open(profile_path, "w") as f:
                    json.dump(params, f, indent=2)
                print(f"✅ Saved disparity settings to default profile: {profile_path}")
            else:
                print("❌ Disparity settings were not saved.")
        except Exception as e:
            print(f"⚠️ Could not save disparity settings to default profile: {e}")


if __name__ == "__main__":
    main()
