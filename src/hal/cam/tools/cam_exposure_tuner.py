# src/hal/cam/cam_exposure_tuner.py
import cv2, os, sys
import numpy as np
from src.hal.cam.Camera import open_stereo_pair

def nothing(x): pass

def show_histogram(winname, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    hist = cv2.normalize(hist, hist).flatten()
    h = 100; w = 256
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(1, 256):
        cv2.line(canvas,
                 (x-1, h-int(hist[x-1]*h)),
                 (x,   h-int(hist[x]*h)),
                 (0,255,0), 1)
    cv2.imshow(winname + " hist", canvas)

def main():
    left_cam, right_cam = open_stereo_pair()

    # Create control window with sliders
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Exposure Left", "Controls", 100, 1000, nothing)
    cv2.createTrackbar("Gain Left", "Controls", 100, 255, nothing)
    cv2.createTrackbar("Exposure Right", "Controls", 100, 1000, nothing)
    cv2.createTrackbar("Gain Right", "Controls", 100, 255, nothing)

    try:
        while True:
            frameL = left_cam.read_frame()
            frameR = right_cam.read_frame()
            if frameL is None or frameR is None:
                continue

            # Apply exposure/gain from trackbars
            expL = cv2.getTrackbarPos("Exposure Left", "Controls")
            gainL = cv2.getTrackbarPos("Gain Left", "Controls")
            expR = cv2.getTrackbarPos("Exposure Right", "Controls")
            gainR = cv2.getTrackbarPos("Gain Right", "Controls")

            # Try to set via v4l2 properties (not all drivers support both)
            left_cam.cap.set(cv2.CAP_PROP_EXPOSURE, float(expL))
            left_cam.cap.set(cv2.CAP_PROP_GAIN, float(gainL))
            right_cam.cap.set(cv2.CAP_PROP_EXPOSURE, float(expR))
            right_cam.cap.set(cv2.CAP_PROP_GAIN, float(gainR))

            # Show streams
            cv2.imshow("Left Camera", frameL)
            cv2.imshow("Right Camera", frameR)

            # Show histograms
            show_histogram("Left Camera", frameL)
            show_histogram("Right Camera", frameR)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        left_cam.close(); right_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
