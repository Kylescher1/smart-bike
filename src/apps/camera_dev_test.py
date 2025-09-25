from hal.Camera import Camera

# create and open a camera
cam = Camera(index=0)
if cam.open():
    ret, frame = cam.read()
    if ret:
        # do something with the frame, e.g., process for obstacle detection
        print(f"Frame shape: {frame.shape}")

cam.close()
