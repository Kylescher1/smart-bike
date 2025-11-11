import os
import sys
import dill
import cv2
import numpy as np
import importlib.util

CONFIG_PATH = r"config.dill"


def load_camera_config():
    print("Loading calibration data...")
    try:
        with open(CONFIG_PATH, "rb") as f:
            config = dill.load(f)
    except Exception as exc:
        raise RuntimeError(f"Unable to load {CONFIG_PATH}: {exc}") from exc

    if "camera" not in config:
        raise KeyError("`camera` section missing from loaded configuration.")

    if "who_to_run" not in config["camera"]:
        raise KeyError("`who_to_run` missing from camera configuration.")

    return config


def resolve_vision_class(camera_config):
    module_path, class_name = camera_config["who_to_run"].rsplit(".", 1)
    module_file = os.path.join(
        os.path.dirname(__file__), *module_path.split(".")
    ) + ".py"

    if not os.path.isfile(module_file):
        raise FileNotFoundError(f"Vision class file not found: {module_file}")

    spec = importlib.util.spec_from_file_location(module_path, module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def check_maps(map_x: np.ndarray, map_y: np.ndarray, size: tuple[int, int], name: str) -> None:
    width, height = size
    expected_shape = (height, width)

    if map_x.shape != expected_shape or map_y.shape != expected_shape:
        raise ValueError(f"{name}: map shape mismatch {map_x.shape}/{map_y.shape} vs {expected_shape}")

    if not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
        raise ValueError(f"{name}: map contains NaN/Inf (wrong projection matrix or size)")

    if (
        map_x.min() < -1
        or map_x.max() > width
        or map_y.min() < -1
        or map_y.max() > height
    ):
        print(
            f"{name}: warning, map samples extend beyond source image "
            f"(min/max x: {map_x.min():.2f}/{map_x.max():.2f}, "
            f"y: {map_y.min():.2f}/{map_y.max():.2f})"
        )


def build_maps_for_size(camera_config: dict, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width, height = size
    left_cfg = camera_config["left"]
    right_cfg = camera_config["right"]

    have_rectify_params = all(
        key in left_cfg and key in right_cfg
        for key in ("K", "D", "R", "P")
    )

    if have_rectify_params:
        K1 = np.asarray(left_cfg["K"], dtype=np.float64)
        D1 = np.asarray(left_cfg["D"], dtype=np.float64)
        R1 = np.asarray(left_cfg["R"], dtype=np.float64)
        P1 = np.asarray(left_cfg["P"], dtype=np.float64)
        K2 = np.asarray(right_cfg["K"], dtype=np.float64)
        D2 = np.asarray(right_cfg["D"], dtype=np.float64)
        R2 = np.asarray(right_cfg["R"], dtype=np.float64)
        P2 = np.asarray(right_cfg["P"], dtype=np.float64)

        newK1 = np.asarray(
            left_cfg.get("newK", P1[:, :3]), dtype=np.float64
        )
        newK2 = np.asarray(
            right_cfg.get("newK", P2[:, :3]), dtype=np.float64
        )

        map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
            K1, D1, R1, newK1, size, cv2.CV_32FC1
        )
        map2x, map2y = cv2.fisheye.initUndistortRectifyMap(
            K2, D2, R2, newK2, size, cv2.CV_32FC1
        )

        check_maps(map1x, map1y, size, "left")
        check_maps(map2x, map2y, size, "right")
        return map1x, map1y, map2x, map2y

    # Fallback to persisted maps (legacy configs)
    if "map_x" not in left_cfg or "map_x" not in right_cfg:
        raise KeyError(
            "Calibration config lacks rectification parameters and persisted maps."
            " Re-run calibrate_maps.py to refresh the dill file."
        )

    map1x = np.ascontiguousarray(left_cfg["map_x"], dtype=np.float32)
    map1y = np.ascontiguousarray(left_cfg["map_y"], dtype=np.float32)
    map2x = np.ascontiguousarray(right_cfg["map_x"], dtype=np.float32)
    map2y = np.ascontiguousarray(right_cfg["map_y"], dtype=np.float32)

    check_maps(map1x, map1y, size, "left")
    check_maps(map2x, map2y, size, "right")
    print("Using legacy maps from config (no rectification matrices stored).")
    return map1x, map1y, map2x, map2y


def build_overlay(left_rect, right_rect, line_spacing=40):
    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    overlay = np.zeros_like(left_rect)
    overlay[..., 2] = left_gray  # Red for left
    overlay[..., 0] = right_gray  # Blue for right

    height = overlay.shape[0]
    for y in range(0, height, line_spacing):
        cv2.line(overlay, (0, y), (overlay.shape[1] - 1, y), (0, 255, 0), 1, cv2.LINE_AA)

    return overlay


def main():
    config = load_camera_config()
    camera_config = config["camera"]

    VisionClass = resolve_vision_class(camera_config)
    vision = VisionClass(name="camera", **camera_config)

    try:
        if not vision.connected:
            print("Starting vision system...")
            vision.start()

        # Prime a frame to discover negotiated size
        left_frame = vision.left_camera.read_frame()
        right_frame = vision.right_camera.read_frame()

        if left_frame is None or right_frame is None:
            raise RuntimeError("Unable to grab initial frames from both cameras.")

        runtime_size_left = (left_frame.shape[1], left_frame.shape[0])
        runtime_size_right = (right_frame.shape[1], right_frame.shape[0])

        if runtime_size_left != runtime_size_right:
            print(f"⚠️ Left/right frame size mismatch: {runtime_size_left} vs {runtime_size_right}")

        runtime_size = runtime_size_left
        print(f"[CamL] {runtime_size_left[0]}x{runtime_size_left[1]}")
        print(f"[CamR] {runtime_size_right[0]}x{runtime_size_right[1]}")
        if camera_config.get("resolution") and tuple(camera_config["resolution"]) != runtime_size:
            print(
                f"⚠️ Stored calibration resolution {tuple(camera_config['resolution'])} "
                f"differs from runtime {runtime_size}. Rebuilding maps for runtime size."
            )

        left_map_x, left_map_y, right_map_x, right_map_y = build_maps_for_size(
            camera_config, runtime_size
        )

        print("Rectification maps ready:")
        print(f"  left map_x: {left_map_x.shape}, dtype={left_map_x.dtype}")
        print(f"  left map_y: {left_map_y.shape}, dtype={left_map_y.dtype}")
        print(f"  right map_x: {right_map_x.shape}, dtype={right_map_x.dtype}")
        print(f"  right map_y: {right_map_y.shape}, dtype={right_map_y.dtype}")

        print("\n" + "=" * 60)
        print("Calibration Verification")
        print("=" * 60)
        print("Press 'q' to quit.")
        print("Press 's' to save the current overlay to disk.")
        print("=" * 60 + "\n")

        os.makedirs("calib_preview", exist_ok=True)
        frame_idx = 0

        pending_left = left_frame
        pending_right = right_frame

        while True:
            if pending_left is not None and pending_right is not None:
                left_frame = pending_left
                right_frame = pending_right
                pending_left = None
                pending_right = None
            else:
                left_frame = vision.left_camera.read_frame()
                right_frame = vision.right_camera.read_frame()

            if left_frame is None or right_frame is None:
                print("⚠️ Failed to grab one or both frames. Retrying...")
                continue

            if left_frame.dtype != np.uint8:
                print(f"Left frame dtype {left_frame.dtype} -> converting to uint8 for display.")
                left_frame = cv2.convertScaleAbs(left_frame)
            if right_frame.dtype != np.uint8:
                print(f"Right frame dtype {right_frame.dtype} -> converting to uint8 for display.")
                right_frame = cv2.convertScaleAbs(right_frame)

            left_rect = cv2.remap(left_frame, left_map_x, left_map_y, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_frame, right_map_x, right_map_y, cv2.INTER_LINEAR)

            overlay = build_overlay(left_rect, right_rect)

            cv2.imshow("Left Rectified", left_rect)
            cv2.imshow("Right Rectified", right_rect)
            cv2.imshow("Stereo Overlay (Left=Red, Right=Blue)", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exiting verification viewer.")
                break
            if key == ord("s"):
                overlay_path = os.path.join("calib_preview", f"overlay_{frame_idx:03d}.png")
                cv2.imwrite(overlay_path, overlay)
                print(f"Saved overlay to {overlay_path}")
                frame_idx += 1

    except KeyboardInterrupt:
        print("\n⚠️ Verification interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        if vision.connected:
            print("Stopping vision system...")
            vision.stop()


if __name__ == "__main__":
    main()


