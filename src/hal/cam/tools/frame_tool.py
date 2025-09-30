import cv2, time

cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

start = time.time()
frames = 0
while frames < 300:  # ~5 seconds at 60fps
    ret, frame = cap.read()
    if not ret:
        break
    frames += 1
end = time.time()

print("Captured", frames, "frames in", round(end-start,2), "seconds")
print("Effective FPS:", round(frames/(end-start),2))

cap.release()
