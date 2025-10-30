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
        # Page 1: Stereo/Disparity core params
        cv2.namedWindow("Disparity Tuner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Disparity Tuner", 420, 520)
        # Page 2: Visualization, Edges, Segmentation
        cv2.namedWindow("Viz/Seg Tuner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Viz/Seg Tuner", 420, 520)
        p = vision._profile_params or {}
        def tb(win, name, maxv, init):
            cv2.createTrackbar(name, win, int(init), int(maxv), lambda v: None)

        # Disparity Tuner window
        tb("Disparity Tuner", "numDisparitiesK", 12, int(p.get("numDisparitiesK", 4)))
        tb("Disparity Tuner", "blockSize", 21, int(p.get("blockSize", 5)))
        tb("Disparity Tuner", "minDisparity", 32, int(p.get("minDisparity", 0)))
        tb("Disparity Tuner", "preFilterCap", 63, int(p.get("preFilterCap", 31)))
        tb("Disparity Tuner", "uniquenessRatio", 40, int(p.get("uniquenessRatio", 10)))
        tb("Disparity Tuner", "speckleWindowSize", 256, int(p.get("speckleWindowSize", 100)))
        tb("Disparity Tuner", "speckleRange", 64, int(p.get("speckleRange", 32)))
        tb("Disparity Tuner", "disp12MaxDiff", 32, int(p.get("disp12MaxDiff", 1)))
        tb("Disparity Tuner", "downSample", 100, int(p.get("downSample", 100)))
        tb("Disparity Tuner", "crop", 200, int(p.get("crop", 0)))

        # Viz/Seg Tuner window
        tb("Viz/Seg Tuner", "farEnhance", 200, int(p.get("farEnhance", 50)))
        tb("Viz/Seg Tuner", "nearCutoff", 255, int(p.get("nearCutoff", 0)))
        tb("Viz/Seg Tuner", "farCutoff", 255, int(p.get("farCutoff", 0)))
        tb("Viz/Seg Tuner", "objThreshMM", 10000, int(p.get("objectThresholdMM", 1500)))
        tb("Viz/Seg Tuner", "colorFocusMM", 20000, int(p.get("colorFocusMM", 3000)))
        tb("Viz/Seg Tuner", "colorSpanMM", 20000, int(p.get("colorSpanMM", 2000)))
        tb("Viz/Seg Tuner", "morphIter", 5, int(p.get("morphIter", 1)))
        tb("Viz/Seg Tuner", "bilateralStrength", 20, int(p.get("bilateralStrength", 8)))
        tb("Viz/Seg Tuner", "wlsLambda", 5000, int(p.get("wlsLambda", 0)))
        tb("Viz/Seg Tuner", "wlsSigma_x10", 100, int(round(float(p.get("wlsSigma", 1.0)) * 10)))
        tb("Viz/Seg Tuner", "useMorph", 1, int(p.get("useMorph", 1)))
        tb("Viz/Seg Tuner", "useBilateral", 1, int(p.get("useBilateral", 1)))
        tb("Viz/Seg Tuner", "useWLS", 1, int(p.get("useWLS", 0)))

        # Edge detection parameters
        tb("Viz/Seg Tuner", "edgeEqualize", 1, int(p.get("edgeEqualize", 1)))
        tb("Viz/Seg Tuner", "edgeBilateralD", 15, int(p.get("edgeBilateralD", 5)))
        tb("Viz/Seg Tuner", "edgeBilateralSigma", 200, int(p.get("edgeBilateralSigma", 60)))
        tb("Viz/Seg Tuner", "edgeCannyKLow_x100", 300, int(round(float(p.get("edgeCannyKLow", 0.66)) * 100)))
        tb("Viz/Seg Tuner", "edgeCannyKHigh_x100", 400, int(round(float(p.get("edgeCannyKHigh", 1.33)) * 100)))
        tb("Viz/Seg Tuner", "edgeUseScharr", 1, int(p.get("edgeUseScharr", 1)))

        # Segmentation/painting mode and params
        tb("Viz/Seg Tuner", "segMode", 2, int(p.get("segMode", 0)))
        tb("Viz/Seg Tuner", "kmK", 10, int(p.get("kmK", 4)))
        tb("Viz/Seg Tuner", "kmSpatialX100", 500, int(p.get("kmSpatialX100", 50)))
        tb("Viz/Seg Tuner", "rgTau", 500, int(p.get("rgTau", 50)))
        tb("Viz/Seg Tuner", "rgSeedStep", 64, int(p.get("rgSeedStep", 16)))
        tb("Viz/Seg Tuner", "wsSigma", 10, int(p.get("wsSigma", 2)))
        tb("Viz/Seg Tuner", "wsMinArea", 5000, int(p.get("wsMinArea", 800)))

    def read_tuner_params() -> dict:
        g = cv2.getTrackbarPos
        params = dict(vision._profile_params or {})
        # Disparity Tuner reads
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

        # Viz/Seg Tuner reads
        params["farEnhance"] = int(g("farEnhance", "Viz/Seg Tuner"))
        params["nearCutoff"] = int(g("nearCutoff", "Viz/Seg Tuner"))
        params["farCutoff"] = int(g("farCutoff", "Viz/Seg Tuner"))
        params["objectThresholdMM"] = int(g("objThreshMM", "Viz/Seg Tuner"))
        params["colorFocusMM"] = int(g("colorFocusMM", "Viz/Seg Tuner"))
        params["colorSpanMM"] = int(g("colorSpanMM", "Viz/Seg Tuner"))
        params["morphIter"] = int(g("morphIter", "Viz/Seg Tuner"))
        params["bilateralStrength"] = int(g("bilateralStrength", "Viz/Seg Tuner"))
        params["wlsLambda"] = int(g("wlsLambda", "Viz/Seg Tuner"))
        params["wlsSigma"] = float(g("wlsSigma_x10", "Viz/Seg Tuner")) / 10.0
        params["useMorph"] = int(g("useMorph", "Viz/Seg Tuner"))
        params["useBilateral"] = int(g("useBilateral", "Viz/Seg Tuner"))
        params["useWLS"] = int(g("useWLS", "Viz/Seg Tuner"))

        # Edge detection parameters
        params["edgeEqualize"] = int(g("edgeEqualize", "Viz/Seg Tuner"))
        params["edgeBilateralD"] = max(1, int(g("edgeBilateralD", "Viz/Seg Tuner")))
        params["edgeBilateralSigma"] = max(0, int(g("edgeBilateralSigma", "Viz/Seg Tuner")))
        params["edgeCannyKLow"] = float(g("edgeCannyKLow_x100", "Viz/Seg Tuner")) / 100.0
        params["edgeCannyKHigh"] = float(g("edgeCannyKHigh_x100", "Viz/Seg Tuner")) / 100.0
        params["edgeUseScharr"] = int(g("edgeUseScharr", "Viz/Seg Tuner"))

        # Segmentation/painting mode and params
        params["segMode"] = int(g("segMode", "Viz/Seg Tuner"))
        params["kmK"] = max(2, int(g("kmK", "Viz/Seg Tuner")))
        params["kmSpatialX100"] = max(0, int(g("kmSpatialX100", "Viz/Seg Tuner")))
        params["rgTau"] = max(1, int(g("rgTau", "Viz/Seg Tuner")))
        params["rgSeedStep"] = max(4, int(g("rgSeedStep", "Viz/Seg Tuner")))
        params["wsSigma"] = max(0, int(g("wsSigma", "Viz/Seg Tuner")))
        params["wsMinArea"] = max(0, int(g("wsMinArea", "Viz/Seg Tuner")))
        return params
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
    def paint_depth_by_regions(edge_img: np.ndarray,
                               depth_map: np.ndarray,
                               invert_colormap: bool = True,
                               norm_low: float | None = None,
                               norm_high: float | None = None,
                               mode: int = 0,
                               params: dict | None = None) -> np.ndarray:
        """
        Create a segmented, "painted" view: flood-fill regions bounded by edges with
        their average depth value (computed from the depth map), then colorize.
        Returns a BGR image sized to the preview window.
        """
        if depth_map is None or depth_map.size == 0:
            return np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)

        h, w = depth_map.shape[:2]
        depth = depth_map.astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0

        painted_depth = np.zeros((h, w), dtype=np.float32)
        mode = int(mode or 0)
        p = params or {}

        if mode == 0:
            # KMeans clustering in (x,y,Z)
            K = max(2, int(p.get("kmK", 4)))
            spw = float(p.get("kmSpatialX100", 50)) / 100.0
            ys, xs = np.mgrid[0:h, 0:w]
            valid = depth > 0
            if not np.any(valid):
                return np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)
            X = np.stack([xs[valid] * spw, ys[valid] * spw, depth[valid]], axis=1).astype(np.float32)
            # kmeans
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            attempts = 1
            flags = cv2.KMEANS_PP_CENTERS
            compactness, labels_km, centers = cv2.kmeans(X, K, None, criteria, attempts, flags)
            labels_km = labels_km.reshape(-1)
            painted_depth = np.zeros((h, w), dtype=np.float32)
            idx = np.flatnonzero(valid)
            for k in range(K):
                mask_idx = idx[labels_km == k]
                if mask_idx.size == 0:
                    continue
                mean_d = float(np.mean(depth.flat[mask_idx]))
                painted_depth.flat[mask_idx] = mean_d

        elif mode == 1:
            # Region growing by depth difference threshold
            tau = float(p.get("rgTau", 50))
            step = max(4, int(p.get("rgSeedStep", 16)))
            labels = -np.ones((h, w), dtype=np.int32)
            cur_label = 0
            for sy in range(0, h, step):
                for sx in range(0, w, step):
                    if labels[sy, sx] != -1 or depth[sy, sx] <= 0:
                        continue
                    cur_label += 1
                    seed_val = depth[sy, sx]
                    q = [(sy, sx)]
                    labels[sy, sx] = cur_label
                    acc = [seed_val]
                    while q:
                        y0, x0 = q.pop()
                        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                            y1, x1 = y0 + dy, x0 + dx
                            if y1 < 0 or y1 >= h or x1 < 0 or x1 >= w:
                                continue
                            if labels[y1, x1] != -1:
                                continue
                            dval = depth[y1, x1]
                            if dval <= 0:
                                continue
                            if abs(dval - seed_val) <= tau:
                                labels[y1, x1] = cur_label
                                acc.append(dval)
                                q.append((y1, x1))
            painted_depth = np.zeros((h, w), dtype=np.float32)
            for lid in range(1, cur_label + 1):
                mask = labels == lid
                if not np.any(mask):
                    continue
                mean_d = float(np.mean(depth[mask]))
                painted_depth[mask] = mean_d

        else:
            # Watershed on depth gradient
            sigma = int(p.get("wsSigma", 2))
            min_area = int(p.get("wsMinArea", 800))
            depth_s = cv2.GaussianBlur(depth, (max(1, 2*sigma+1), max(1, 2*sigma+1)), 0) if sigma > 0 else depth
            gx = cv2.Sobel(depth_s, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(depth_s, cv2.CV_32F, 0, 1, ksize=3)
            gmag = cv2.magnitude(gx, gy)
            g8 = cv2.normalize(gmag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # sure background as high gradient, sure foreground as eroded valid low gradient
            _, edges_bin = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            valid = (depth > 0).astype(np.uint8) * 255
            sure_fg = cv2.morphologyEx(valid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=1)
            dist = cv2.distanceTransform(sure_fg, cv2.DIST_L2, 3)
            _, markers = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
            markers = markers.astype(np.uint8)
            num, markers_cc = cv2.connectedComponents(markers)
            markers_cc = markers_cc + 1
            unknown = cv2.subtract(valid, sure_fg)
            markers_cc[unknown == 255] = 0
            # Need 3-channel image; use colorized gradient as placeholder
            g3 = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
            cv2.watershed(g3, markers_cc)
            labels = markers_cc
            painted_depth = np.zeros((h, w), dtype=np.float32)
            for lid in np.unique(labels):
                if lid <= 1:
                    continue
                mask = labels == lid
                area = int(mask.sum())
                if area < min_area:
                    continue
                dvals = depth[mask]
                dvals = dvals[(dvals > 0) & np.isfinite(dvals)]
                if dvals.size == 0:
                    continue
                painted_depth[mask] = float(np.mean(dvals))

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
                painted = paint_depth_by_regions(edge_map,
                                                depth_vis,
                                                invert_colormap=True,
                                                norm_low=low,
                                                norm_high=high,
                                                mode=int(vision._profile_params.get("segMode", 0)),
                                                params=vision._profile_params)
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
