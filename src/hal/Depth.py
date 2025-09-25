import cv2
from Camera import Camera  # <- match your filename (camera.py)

def main():
    # Open left and right cameras
    left_cam = Camera(index=3)
    right_cam = Camera(index=1)

    if not left_cam.open() or not right_cam.open():
        print("❌ Could not open both cameras.")
        return

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

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting...")
                break

    finally:
        left_cam.close()
        right_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
