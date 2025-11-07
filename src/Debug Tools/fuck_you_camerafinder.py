import cv2


def find_camera_indices():
    available_indices = []
    index = 0
    while True:
        print(f"Try camera{index}")
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"{index} cannont be opened, go cry")
            # If the camera cannot be opened, it likely means this index is invalid
            # or there are no more cameras.
            break
        print(f"{index} was  opened, go cry anyways")
        # Try to read a frame to confirm the camera is truly active
        ret, frame = cap.read()
        if ret:
            available_indices.append(index)

        cap.release()  # Release the camera
        index += 1
    return available_indices


camera_indices = find_camera_indices()
if camera_indices:
    print(f"Found camera indices: {camera_indices}")
else:
    print("No cameras found.")