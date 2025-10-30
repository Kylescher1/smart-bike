"""Entry point for the Smart Bike vision pipeline."""
from __future__ import annotations

import time

import cv2
import numpy as np

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
        tb("farCutoff", 255, int(p.get("farCutoff", 0)))
        tb("objThreshMM", 10000, int(p.get("objectThresholdMM", 1500)))
        # Color mapping focus controls
        tb("colorFocusMM", 20000, int(p.get("colorFocusMM", 3000)))
        tb("colorSpanMM", 20000, int(p.get("colorSpanMM", 2000)))
        tb("morphIter", 5, int(p.get("morphIter", 1)))
        tb("bilateralStrength", 20, int(p.get("bilateralStrength", 8)))
        tb("wlsLambda", 5000, int(p.get("wlsLambda", 0)))
        tb("wlsSigma_x10", 100, int(round(float(p.get("wlsSigma", 1.0)) * 10)))
        # binary toggles as 0/1
        tb("useMorph", 1, int(p.get("useMorph", 1)))
        tb("useBilateral", 1, int(p.get("useBilateral", 1)))
        tb("useWLS", 1, int(p.get("useWLS", 0)))

        # Edge detection parameters
        tb("edgeEqualize", 1, int(p.get("edgeEqualize", 1)))
        tb("edgeBilateralD", 15, int(p.get("edgeBilateralD", 5)))
        tb("edgeBilateralSigma", 200, int(p.get("edgeBilateralSigma", 60)))
        tb("edgeCannyKLow_x100", 300, int(round(float(p.get("edgeCannyKLow", 0.66)) * 100)))
        tb("edgeCannyKHigh_x100", 400, int(round(float(p.get("edgeCannyKHigh", 1.33)) * 100)))
        tb("edgeUseScharr", 1, int(p.get("edgeUseScharr", 1)))

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
        params["farCutoff"] = int(g("farCutoff", "Disparity Tuner"))
        params["objectThresholdMM"] = int(g("objThreshMM", "Disparity Tuner"))
        # Color mapping focus controls
        params["colorFocusMM"] = int(g("colorFocusMM", "Disparity Tuner"))
        params["colorSpanMM"] = int(g("colorSpanMM", "Disparity Tuner"))
        params["morphIter"] = int(g("morphIter", "Disparity Tuner"))
        params["bilateralStrength"] = int(g("bilateralStrength", "Disparity Tuner"))
        params["wlsLambda"] = int(g("wlsLambda", "Disparity Tuner"))
        params["wlsSigma"] = float(g("wlsSigma_x10", "Disparity Tuner")) / 10.0
        params["useMorph"] = int(g("useMorph", "Disparity Tuner"))
        params["useBilateral"] = int(g("useBilateral", "Disparity Tuner"))
        params["useWLS"] = int(g("useWLS", "Disparity Tuner"))

        # Edge detection parameters
        params["edgeEqualize"] = int(g("edgeEqualize", "Disparity Tuner"))
        params["edgeBilateralD"] = max(1, int(g("edgeBilateralD", "Disparity Tuner")))
        params["edgeBilateralSigma"] = max(0, int(g("edgeBilateralSigma", "Disparity Tuner")))
        params["edgeCannyKLow"] = float(g("edgeCannyKLow_x100", "Disparity Tuner")) / 100.0
        params["edgeCannyKHigh"] = float(g("edgeCannyKHigh_x100", "Disparity Tuner")) / 100.0
        params["edgeUseScharr"] = int(g("edgeUseScharr", "Disparity Tuner"))
        return params

    create_tuner_window()

    # Ensure the Depth map window is 1920x1080
    cv2.namedWindow("Depth map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth map", 1920, 1080)

    # --- Pause/inspect state ---
    paused = False
    pause_depth = None  # original depth map (float)
    pause_display = None  # 1920x1080 BGR shown image
    clicked_points = []  # list of (x,y,depth)
    DISP_W, DISP_H = 1920, 1080

    def on_depth_click(event, x, y, flags, param):
        nonlocal clicked_points, pause_depth, pause_display
        if not paused or pause_depth is None or pause_display is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            h0, w0 = pause_depth.shape[:2]
            # Map display coords (1920x1080) back to depth map coords
            x0 = int(round(x * (w0 / float(DISP_W))))
            y0 = int(round(y * (h0 / float(DISP_H))))
            x0 = max(0, min(w0 - 1, x0))
            y0 = max(0, min(h0 - 1, y0))
            d = float(pause_depth[y0, x0])
            clicked_points.append((x, y, d))
            # Also print to console
            print(f"Clicked depth at ({x0},{y0}) -> {d:.3f}")

    cv2.setMouseCallback("Depth map", on_depth_click)

    # --- Region painting utility ---
    def paint_depth_by_regions(edge_img: np.ndarray, depth_map: np.ndarray, invert_colormap: bool = True, norm_low: float | None = None, norm_high: float | None = None) -> np.ndarray:
        """
        Create a segmented, "painted" view: flood-fill regions bounded by edges with
        their average depth value (computed from the depth map), then colorize.
        Returns a BGR image sized to the preview window.
        """
        if depth_map is None or depth_map.size == 0:
            return np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)

        # Ensure single-channel binary edges
        edges = edge_img
        if edges.ndim == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        _, edges_bin = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)

        # Resize edges to match depth map resolution
        h, w = depth_map.shape[:2]
        edges_bin = cv2.resize(edges_bin, (w, h), interpolation=cv2.INTER_NEAREST)

        # Add depth-discontinuity boundaries to help close silhouettes
        depth = depth_map.astype(np.float32).copy()
        depth[~np.isfinite(depth)] = 0.0
        # Smooth depth to reduce noise before gradient
        depth_s = cv2.GaussianBlur(depth, (5, 5), 0)
        gx = cv2.Sobel(depth_s, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_s, cv2.CV_32F, 0, 1, ksize=3)
        gmag = cv2.magnitude(gx, gy)
        # Normalize gradient for thresholding
        if np.any(gmag > 0):
            gnorm = cv2.normalize(gmag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # Otsu threshold to pick strong depth edges
            _, depth_edge = cv2.threshold(gnorm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            depth_edge = np.zeros_like(edges_bin)

        # Combine image edges with depth edges and thicken to better seal regions
        boundaries = cv2.bitwise_or(edges_bin, depth_edge)
        edges_thick = cv2.dilate(boundaries, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

        # Valid fillable pixels: not an edge and have valid positive depth
        valid = (edges_thick == 0) & (depth_map > 0)
        if not np.any(valid):
            return np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)

        valid_u8 = (valid.astype(np.uint8) * 255)
        # Remove tiny holes/gaps by closing before labeling
        valid_u8 = cv2.morphologyEx(valid_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        # Connected components on valid regions
        num_labels, labels = cv2.connectedComponents(valid_u8, connectivity=4)

        h, w = depth_map.shape[:2]
        painted_depth = np.zeros((h, w), dtype=np.float32)

        # Compute mean depth per region; skip label 0 (background). Ignore tiny regions.
        min_region_area = max(200, int(0.001 * h * w))
        for label_id in range(1, num_labels):
            mask = (labels == label_id)
            area = int(mask.sum())
            if area < min_region_area:
                continue
            dvals = depth_map[mask]
            dvals = dvals[np.isfinite(dvals) & (dvals > 0)]
            if dvals.size == 0:
                continue
            mean_depth = float(dvals.mean())
            painted_depth[mask] = mean_depth

        # Normalize and colorize like the main depth view
        if not np.any(painted_depth):
            vis = np.zeros((h, w), dtype=np.uint8)
        else:
            if norm_low is not None and norm_high is not None and norm_high > norm_low:
                scaled = np.clip((painted_depth - norm_low) / float(norm_high - norm_low), 0.0, 1.0)
                norm = (scaled * 255.0).astype(np.uint8)
            else:
                norm = cv2.normalize(painted_depth, None, 0, 255, cv2.NORM_MINMAX)
            if invert_colormap:
                norm = 255 - norm
            vis = norm.astype(np.uint8)
        color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        return cv2.resize(color, (DISP_W, DISP_H), interpolation=cv2.INTER_AREA)

    try:
        while True:
            # --- performance tracker ---
            times = {}
            t_prev = time.perf_counter()
            def _mark(label: str):
                nonlocal t_prev, times
                t_now = time.perf_counter()
                times[label] = (t_now - t_prev) * 1000.0
                t_prev = t_now
            if not paused:
                frames = vision.capture_frames()
                if frames is None:
                    continue

                left_frame, right_frame = frames
                _mark("capture")
                # Update live params from tuner
                vision._profile_params = read_tuner_params()
                # Update close-object threshold from settings
                try:
                    vision.object_distance_threshold_mm = float(vision._profile_params.get("objectThresholdMM", vision.object_distance_threshold_mm))
                except Exception:
                    pass
                _mark("read_tuner")
                depth_result = vision.compute_depth(left_frame, right_frame)
                _mark("compute_depth")

            # --- Primary edges from depth map (3D boundaries) ---
            # Fallback to image edges only if depth not available this frame
            edge_map = None
            if not paused:
                try:
                    dv = depth_result.depth_map.astype(np.float32)
                    dv[np.isnan(dv) | np.isinf(dv)] = 0.0
                    if dv.size:
                        dv_blur = cv2.GaussianBlur(dv, (5, 5), 0)
                        gx = cv2.Sobel(dv_blur, cv2.CV_32F, 1, 0, ksize=3)
                        gy = cv2.Sobel(dv_blur, cv2.CV_32F, 0, 1, ksize=3)
                        gmag = cv2.magnitude(gx, gy)
                        if np.any(gmag > 0):
                            gnorm = cv2.normalize(gmag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            _, edge_map = cv2.threshold(gnorm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            edge_map = cv2.dilate(edge_map, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
                except Exception:
                    edge_map = None

            if edge_map is None:
                # Fallback: Edge detection on original (non-depth) image
                gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                if int(vision._profile_params.get("edgeEqualize", 1)):
                    gray = cv2.equalizeHist(gray)
                d_edge = max(1, int(vision._profile_params.get("edgeBilateralD", 5)))
                s_edge = max(0, int(vision._profile_params.get("edgeBilateralSigma", 60)))
                denoised = cv2.bilateralFilter(gray, d_edge, s_edge, s_edge)
                v = float(np.median(denoised))
                k_low = float(vision._profile_params.get("edgeCannyKLow", 0.66))
                k_high = float(vision._profile_params.get("edgeCannyKHigh", 1.33))
                lower = int(max(0, k_low * v))
                upper = int(min(255, k_high * v))
                edges_canny = cv2.Canny(denoised, lower, upper, L2gradient=True)
                if int(vision._profile_params.get("edgeUseScharr", 1)):
                    gx = cv2.Scharr(denoised, cv2.CV_32F, 1, 0)
                    gy = cv2.Scharr(denoised, cv2.CV_32F, 0, 1)
                    mag = cv2.magnitude(gx, gy)
                    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    _, edges_scharr = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    edge_map = cv2.bitwise_or(edges_canny, edges_scharr)
                else:
                    edge_map = edges_canny
            _mark("edges")

            if not paused:
                depth_vis = depth_result.depth_map
                depth_color = None
                if depth_vis.size:
                    # Focused color mapping window based on sliders
                    focus_mm = float(vision._profile_params.get("colorFocusMM", 3000))
                    span_mm = float(vision._profile_params.get("colorSpanMM", 2000))
                    low = high = None
                    d = depth_vis.astype("float32")
                    valid = d[np.isfinite(d) & (d > 0)]
                    if valid.size and span_mm > 0:
                        half = span_mm / 2.0
                        low = max(float(valid.min()), focus_mm - half)
                        high = min(float(valid.max()), focus_mm + half)
                        if high <= low:
                            low = float(valid.min())
                            high = float(valid.max())
                    if low is None or high is None or high <= low:
                        # fallback to global min-max
                        low = float(valid.min()) if valid.size else 0.0
                        high = float(valid.max()) if valid.size else 1.0
                        if high <= low:
                            high = low + 1.0

                    scaled = np.clip((d - low) / float(high - low), 0.0, 1.0)
                    norm = (scaled * 255.0).astype(np.uint8)
                    norm = 255 - norm  # invert so nearer appears warmer
                    depth_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                    # Resize to display size and keep originals for pause mode
                    pause_display = cv2.resize(depth_color, (DISP_W, DISP_H), interpolation=cv2.INTER_AREA)
                    pause_depth = depth_vis.copy()
                    _mark("colorize_resize")
            else:
                depth_color = pause_display.copy() if pause_display is not None else None
                # Draw clicked points with depth labels on the paused display
                if depth_color is not None and clicked_points:
                    for (cx, cy, dval) in clicked_points:
                        cv2.circle(depth_color, (int(cx), int(cy)), 5, (0, 255, 255), 2)
                        label = f"{dval:.3f}"
                        cv2.putText(depth_color, label, (int(cx) + 8, int(cy) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                _mark("paused_draw")

            shown = False
            if depth_color is not None:
                cv2.imshow("Depth map", depth_color)
                shown = True
            # Show painted depth view based on depth-derived edges (use same color window)
            if not paused and depth_vis.size:
                painted = paint_depth_by_regions(edge_map, depth_vis, invert_colormap=True, norm_low=low, norm_high=high)
                cv2.imshow("Painted depth", painted)
                shown = True
            _mark("imshow")
            if shown:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('p'):
                    paused = not paused
                    if paused:
                        clicked_points = []
                        print("⏸️  Paused. Click on the depth view to sample depths. Press 'p' to resume.")
                    else:
                        clicked_points = []
                        print("▶️  Resumed. Clearing sampled points.")
                elif key == ord("q"):
                    break
            # Print timing summary for this iteration
            if times:
                total_ms = sum(times.values())
                parts = " | ".join(f"{k}:{v:.1f}ms" for k, v in times.items())
                fps = 1000.0 / total_ms if total_ms > 0 else 0.0
                print(f"{parts} | total:{total_ms:.1f}ms | fps:{fps:.1f}")

            # Object detection based on nearest sufficiently large region mean depth
            try:
                if not paused and depth_vis.size:
                    edges_src = edge_map if edge_map.ndim == 2 else cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)
                    _, edges_bin = cv2.threshold(edges_src, 0, 255, cv2.THRESH_BINARY)
                    edges_thick = cv2.dilate(edges_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
                    valid = (edges_thick == 0) & (depth_vis > 0)
                    valid_u8 = (valid.astype(np.uint8) * 255)
                    num_labels, labels = cv2.connectedComponents(valid_u8, connectivity=4)
                    h, w = depth_vis.shape[:2]
                    min_region_area = max(100, int(0.0005 * h * w))
                    nearest = None
                    for lid in range(1, num_labels):
                        mask = (labels == lid)
                        area = int(mask.sum())
                        if area < min_region_area:
                            continue
                        dvals = depth_vis[mask]
                        dvals = dvals[np.isfinite(dvals) & (dvals > 0)]
                        if dvals.size == 0:
                            continue
                        mean_d = float(dvals.mean())
                        if nearest is None or mean_d < nearest:
                            nearest = mean_d
                    if nearest is not None and nearest <= vision.object_distance_threshold_mm:
                        vision.warn_rider()
            except Exception:
                pass

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
