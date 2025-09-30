# main.py
from src.hal.cam.Camera import Camera
from src.hal.cam.depth import compute_depth_map
import cv2
import time

def main():
    cameras = []
    for idx in [1, 3]:
        try:
            cam = Camera(index=idx)
            cam.open()
            cameras.append(cam)
        except RuntimeError as e:
            print(e)

    if not cameras:
        print("❌ No cameras available. Exiting.")
        return

    print(f"✅ Running with cameras {', '.join(str(c.index) for c in cameras)}")

    try:
        while True:
            for cam in cameras:
                depth_map = compute_depth_map(cam)
                if depth_map is not None:
                    cv2.imshow(f"Depth Map {cam.index}", depth_map)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)
    finally:
        for cam in cameras:
            cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
