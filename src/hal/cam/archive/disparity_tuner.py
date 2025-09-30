import cv2
import numpy as np
import time
from Camera import Camera

def nothing(x):
    pass

def main():
    # Open stereo cameras
    left_cam = Camera(index=3)
    right_cam = Camera(index=1)

    if not left_cam.open() or not right_cam.open():
        print("❌ Could not open both cameras.")
        return

    # Create a window with trackbars
    cv2.namedWindow("DisparityBM", cv2.WINDOW_NORMAL)

    # Trackbars for parameters
    cv2.createTrackbar("numDisp", "DisparityBM", 8, 16, nothing)      # *16
    cv2.createTrackbar("blockSize", "DisparityBM", 15, 51, nothing)   # odd only
    cv2.createTrackbar("minDisp", "DisparityBM", 0, 50, nothing)
    cv2.createTrackbar("textureThresh", "DisparityBM", 10, 100, nothing)
    cv2.createTrackbar("uniquenessRatio", "DisparityBM", 15, 50, nothing)
    cv2.createTrackbar("speckleWin", "DisparityBM", 50, 300, nothing)
    cv2.createTrackbar("speckleRange", "DisparityBM", 2, 20, nothing)
    cv2.createTrackbar("preFilterSize", "DisparityBM", 9, 31, nothing) # odd only
    cv2.createTrackbar("preFilterCap", "DisparityBM", 31, 63, nothing)

    last_update = 0
    try:
        while True:
            retL, frameL = left_cam.read()
            retR, frameR = right_cam.read()

            if not (retL and retR):
                print("⚠️ Frame grab failed.")
                continue

            grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

            # Update disparity map once per second
            if time.time() - last_update > 1:
                numDisp = cv2.getTrackbarPos("numDisp", "DisparityBM") * 16
                if numDisp < 16: numDisp = 16

                blockSize = cv2.getTrackbarPos("blockSize", "DisparityBM")
                if blockSize < 5:
                    blockSize = 5
                if blockSize % 2 == 0:
                    blockSize += 1
                if blockSize > min(grayL.shape[:2]):  # not bigger than image size
                    blockSize = min(grayL.shape[:2]) | 1  # force odd


                minDisp = cv2.getTrackbarPos("minDisp", "DisparityBM")
                textureThresh = cv2.getTrackbarPos("textureThresh", "DisparityBM")
                uniquenessRatio = cv2.getTrackbarPos("uniquenessRatio", "DisparityBM")
                speckleWin = cv2.getTrackbarPos("speckleWin", "DisparityBM")
                speckleRange = cv2.getTrackbarPos("speckleRange", "DisparityBM")
                preFilterSize = cv2.getTrackbarPos("preFilterSize", "DisparityBM")
                if preFilterSize % 2 == 0: preFilterSize += 1  # must be odd
                preFilterCap = cv2.getTrackbarPos("preFilterCap", "DisparityBM")

                stereo = cv2.StereoBM_create(numDisparities=numDisp, blockSize=blockSize)
                stereo.setMinDisparity(minDisp)
                stereo.setTextureThreshold(textureThresh)
                stereo.setUniquenessRatio(uniquenessRatio)
                stereo.setSpeckleWindowSize(speckleWin)
                stereo.setSpeckleRange(speckleRange)
                stereo.setPreFilterSize(preFilterSize)
                stereo.setPreFilterCap(preFilterCap)

                disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
                disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
                disp_vis = np.uint8(disp_vis)

                cv2.imshow("DisparityBM", disp_vis)
                last_update = time.time()

            # Show live input for reference
            cv2.imshow("Left", frameL)
            cv2.imshow("Right", frameR)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        left_cam.close()
        right_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
