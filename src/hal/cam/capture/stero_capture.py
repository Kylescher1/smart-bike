import cv2
import os
from Camera import Camera  # make sure camera.py is in the same folder

def main():
    # Create output folder for stereo pairs
    save_dir = "stereo_pairs"
    os.makedirs(save_dir, exist_ok=True)

    # Open left and right cameras
    left_cam = Camera(index=3)
    right_cam = Camera(index=1)

    if not left_cam.open() or not right_cam.open():
        print("❌ Could not open both cameras.")
        return

    pair_count = 0  # counter for saved pairs

    try:
        while True:
            retL, frameL = left_cam.read()
            retR, frameR = right_cam.read()

            if not (retL and retR):
                print("⚠️ Failed to grab one or both frames.")
                continue

            # Show each camera in a separate window
            cv2.imshow("Left Camera", frameL)
            cv2.imshow("Right Camera", frameR)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):  # quit
                print("Exiting...")
                break

            elif key == ord("s"):  # save stereo pair
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
