import cv2

cap1 = cv2.VideoCapture(1, cv2.CAP_V4L2)  # Change to the correct index (1 or 2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)  # camera 2

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Couldn't access one or both of cameras.")
else:
    print("Cameras opened successfully.")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1:
            cv2.imshow("Camera 1", frame1)

        if ret2:
            cv2.imshow("Camera 2", frame2)
        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

# Release the capture object and close all windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
