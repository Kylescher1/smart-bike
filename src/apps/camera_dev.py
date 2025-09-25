import cv2

def find_cameras(max_index=10):
    """Scan through indices and return opened VideoCapture objects."""
    cameras = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            cameras.append((i, cap))
        else:
            cap.release()
    return cameras


def main():
    cameras = find_cameras(max_index=10)

    if not cameras:
        print("No cameras detected.")
        return

    print(f"{len(cameras)} camera(s) opened successfully.")

    while True:
        for idx, cap in cameras:
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Camera {idx}", frame)

        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release all opened cameras
    for _, cap in cameras:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
