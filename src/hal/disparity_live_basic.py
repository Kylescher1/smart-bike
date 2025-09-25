import cv2
import numpy as np

def nothing(x):
    pass

def main():
    # Load fisheye calibration
    calib = np.load("stereo_calib_fisheye.npz")
    K1, D1 = calib["K1"], calib["D1"]
    K2, D2 = calib["K2"], calib["D2"]
    R1, R2 = calib["R1"], calib["R2"]
    P1, P2 = calib["P1"], calib["P2"]

    # Open stereo cameras
    left_cam = cv2.VideoCapture(3)
    right_cam = cv2.VideoCapture(1)

    if not left_cam.isOpened() or not right_cam.isOpened():
        print("❌ Could not open cameras")
        return

    # Grab one frame to get resolution
    retL, frameL = left_cam.read()
    retR, frameR = right_cam.read()
    if not (retL and retR):
        print("❌ Could not grab initial frames")
        return

    h, w = frameL.shape[:2]

    # Build rectification maps
    mapLx, mapLy = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    mapRx, mapRy = cv2.fisheye.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    # Create window + trackbars
    cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("numDisp", "Disparity", 8, 16, nothing)      # *16
    cv2.createTrackbar("blockSize", "Disparity", 5, 21, nothing)    # must be odd
    cv2.createTrackbar("uniqueness", "Disparity", 10, 30, nothing)
    cv2.createTrackbar("speckleWin", "Disparity", 50, 200, nothing)
    cv2.createTrackbar("speckleRange", "Disparity", 2, 10, nothing)

    print("✅ Running live disparity with tuner. Press 'q' to quit.")

    while True:
        retL, frameL = left_cam.read()
        retR, frameR = right_cam.read()
        if not (retL and retR):
            print("⚠️ Frame grab failed.")
            continue

        # Rectify
        rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

        # Convert to grayscale
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

        # Read parameters from trackbars
        num_disp = cv2.getTrackbarPos("numDisp", "Disparity") * 16
        if num_disp < 16: num_disp = 16

        block_size = cv2.getTrackbarPos("blockSize", "Disparity")
        if block_size < 3: block_size = 3
        if block_size % 2 == 0: block_size += 1  # must be odd

        uniqueness = cv2.getTrackbarPos("uniqueness", "Disparity")
        speckleWin = cv2.getTrackbarPos("speckleWin", "Disparity")
        speckleRange = cv2.getTrackbarPos("speckleRange", "Disparity")

        # Stereo matcher (SGBM)
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            uniquenessRatio=uniqueness,
            speckleWindowSize=speckleWin,
            speckleRange=speckleRange,
            disp12MaxDiff=1
        )

        disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)

        # Show images
        cv2.imshow("Left Rectified", rectL)
        cv2.imshow("Right Rectified", rectR)
        cv2.imshow("Disparity", disp_vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    left_cam.release()
    right_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
