import cv2

cap = cv2.VideoCapture(1, cv2.CAP_V4L2)  # Change to the correct index (1 or 2)

if not cap.isOpened():
    print("Error: Couldn't access the camera.")
else:
    print("Camera opened successfully.")

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Camera Test", frame)

        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

# Release the capture object and close all windows
cap.release()
cv2.destroyAllWindows()
