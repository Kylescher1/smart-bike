# src/hal/cam/calibrate/stero_capture.py
import cv2
import sys, os
import glob

# Add repo root (3 levels up from /src/hal/cam/calibrate)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from src.hal.cam.Camera import open_stereo_pair

def main():
    base_dir = os.path.dirname(__file__)          # /src/hal/cam/calibrate
    save_dir = os.path.join(base_dir, "stereo_pairs")
    os.makedirs(save_dir, exist_ok=True)

    # Clear out existing files
    for f in glob.glob(os.path.join(save_dir, "*.png")):
        try:
            os.remove(f)
        except OSError as e:
            print(f"⚠️ Could not delete {f}: {e}")
    print(f"🧹 Cleared out existing files in {save_dir}")

    try:
        left_cam, right_cam = open_stereo_pair()   # automatically picks two working cameras
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

            previewL = cv2.resize(frameL, (1024, 960))
            previewR = cv2.resize(frameR, (1024, 960))
            cv2.imshow("Left Camera", previewL)
            cv2.imshow("Right Camera", previewR)


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
