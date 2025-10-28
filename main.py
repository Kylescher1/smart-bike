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

    try:
        while True:
            frames = vision.capture_frames()
            if frames is None:
                continue

            left_frame, right_frame = frames
            depth_result = vision.compute_depth(left_frame, right_frame)
            edge_map = VisionSystem.edge_map_from_depth(depth_result.depth_map)
            depth_vis = depth_result.depth_map
            depth_color = None
            if depth_vis.size:
                norm = cv2.normalize(depth_vis.astype("float32"), None, 0, 255, cv2.NORM_MINMAX)
                depth_color = cv2.applyColorMap(norm.astype("uint8"), cv2.COLORMAP_JET)

            shown = False
            if depth_color is not None:
                cv2.imshow("Depth map", depth_color)
                shown = True
            if edge_map.size:
                cv2.imshow("Depth edges", edge_map)
                shown = True
            if shown:
                if cv2.waitKey(1) & 0xFF == ord("q"):
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


if __name__ == "__main__":
    main()
