from src.hal.cam.Camera import open_stereo_pair
from src.hal.cam.depth import load_calibration, compute_depth_map
import cv2, time

def main():
    calib = load_calibration()
    try:
        left_cam, right_cam = open_stereo_pair()
    except RuntimeError as e:
        print(e)
        return

    print(f"✅ Running stereo with cameras {left_cam.index}, {right_cam.index}")

    try:
        while True:
            left_frame = left_cam.read_frame()
            right_frame = right_cam.read_frame()
            if left_frame is None or right_frame is None:
                continue

            depth_map = compute_depth_map(left_frame, right_frame, calib)
            cv2.imshow("Depth Map", depth_map)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)
    finally:
        left_cam.close(); right_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
