import cv2
import numpy as np
from Camera import Camera  # your wrapper

def load_calibration(filename="stereo_calib_fisheye.npz"):
    data = np.load(filename, allow_pickle=True)
    return (data["leftMapX"], data["leftMapY"],
            data["rightMapX"], data["rightMapY"],
            tuple(data["imageSize"]), data["Q"])

def draw_epilines(imgL, imgR, n_lines=15):
    h, w, _ = imgL.shape
    stacked = np.hstack((imgL, imgR))
    step = h // n_lines
    for y in range(0, h, step):
        cv2.line(stacked, (0, y), (2*w, y), (0, 255, 0), 1)
    return stacked

def nothing(x): pass

def create_tuner_window():
    cv2.namedWindow("Tuner")

    # numDisparities must be divisible by 16, so we’ll multiply slider value * 16
    cv2.createTrackbar("numDisparities", "Tuner", 4, 20, nothing)  # -> 64..320
    cv2.createTrackbar("blockSize", "Tuner", 5, 21, nothing)       # odd only
    cv2.createTrackbar("uniquenessRatio", "Tuner", 15, 50, nothing)
    cv2.createTrackbar("speckleWindowSize", "Tuner", 50, 200, nothing)
    cv2.createTrackbar("speckleRange", "Tuner", 2, 10, nothing)
    cv2.createTrackbar("disp12MaxDiff", "Tuner", 1, 25, nothing)

def get_stereo_matcher():
    # Read values from trackbars
    numDisp = cv2.getTrackbarPos("numDisparities", "Tuner") * 16
    if numDisp < 16: numDisp = 16

    blockSize = cv2.getTrackbarPos("blockSize", "Tuner")
    if blockSize % 2 == 0: blockSize += 1  # must be odd, >1
    if blockSize < 3: blockSize = 3

    uniq = cv2.getTrackbarPos("uniquenessRatio", "Tuner")
    speckleWS = cv2.getTrackbarPos("speckleWindowSize", "Tuner")
    speckleRange = cv2.getTrackbarPos("speckleRange", "Tuner")
    disp12MaxDiff = cv2.getTrackbarPos("disp12MaxDiff", "Tuner")

    # Derive penalties
    P1 = 8 * 1 * blockSize**2
    P2 = 32 * 1 * blockSize**2

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numDisp,
        blockSize=blockSize,
        P1=P1, P2=P2,
        disp12MaxDiff=disp12MaxDiff,
        uniquenessRatio=uniq,
        speckleWindowSize=speckleWS,
        speckleRange=speckleRange,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return stereo

def main():
    leftMapX, leftMapY, rightMapX, rightMapY, imageSize, Q = load_calibration()

    # Open stereo cameras
    camL = Camera(index=3, width=imageSize[0], height=imageSize[1])
    camR = Camera(index=1, width=imageSize[0], height=imageSize[1])
    if not camL.open() or not camR.open():
        print("❌ Could not open both cameras")
        return

    create_tuner_window()

    print("🎥 Adjust sliders in 'Tuner' window. Press ESC to exit.")
    try:
        while True:
            retL, frameL = camL.read()
            retR, frameR = camR.read()
            if not retL or not retR:
                print("❌ Frame capture failed")
                break

            # Rectify
            rectL = cv2.remap(frameL, leftMapX, leftMapY, cv2.INTER_LINEAR)
            rectR = cv2.remap(frameR, rightMapX, rightMapY, cv2.INTER_LINEAR)

            grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

            stereo = get_stereo_matcher()
            disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
            disparity[disparity < 0] = 0  # mask invalid

            # Debug values
            print("min disp:", np.min(disparity), "max disp:", np.max(disparity))

            # Debug values
            print("min disp:", np.min(disparity), "max disp:", np.max(disparity))

            # Normalize and grayscale
            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = np.uint8(disp_vis)

            cv2.imshow("Disparity (Grayscale)", disp_vis)


            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
    finally:
        camL.close()
        camR.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
