import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found")
    exit()

# Read one frame to initialize background
ret, background = cap.read()
if not ret:
    print("Failed to grab frame")
    exit()

background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Absolute difference with initial frame
    delta = cv2.absdiff(background_gray, gray)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Count nonzero pixels to estimate motion
    motion = np.sum(thresh) / 255
    if motion > 10000:  # adjust threshold
        text = "Object detected ahead"
        color = (0, 0, 255)
    else:
        text = "Clear"
        color = (0, 255, 0)

    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("View", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
