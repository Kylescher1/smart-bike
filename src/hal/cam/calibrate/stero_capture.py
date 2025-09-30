import cv2
import sys, os

# Add repo root (3 levels up from /src/hal/cam/calibrate)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from src.hal.cam.Camera import Camera

def main():
    base_dir = os.path.dirname(__file__)          # /src/hal/cam/calibrate
    save_dir = os.path.join(base_dir, "stereo_pairs")
    os.makedirs(save_dir, exist_ok=True)

    left_cam = Camera(index=3)
    right_cam = Camera(index=1)

    try:
        left_cam.open()
        right_cam.open()
    except RuntimeError as e:
        print(e)
        return

    pair_count = 0

    try:
        while True:
            frameL = left_cam.read_frame()
            frameR = right_cam.read_frame()

            if frameL is None or frameR is None:
                print("⚠️ Failed to grab one or both frames.")
                continue

            cv2.imshow("Left Camera", frameL)
            cv2.imshow("Right Camera", frameR)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exiting...")
                break
            elif key == ord("s"):
                left_name = os.path.join(save_dir, f"left_{pair_count:03d}.png")
                right_name = os.path.join(save_dir, f"right_{pair_count:03d}.png")
                cv2.imwrite(left_name, frameL)
                cv2.imwrite(right_name, frameR)
                print(f"💾 Saved stereo pair: {left_name}, {right_name}")
                pair_count += 1
    finally:
        left_cam.close()
        right_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
