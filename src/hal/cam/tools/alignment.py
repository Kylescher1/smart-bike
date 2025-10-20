import cv2, glob, os

base_dir = os.path.dirname(__file__)                     # /cam/tools
pairs_dir = os.path.join(base_dir, "../calibrate/stereo_pairs")
pairs_dir = os.path.abspath(pairs_dir)

pairs = list(zip(sorted(glob.glob(os.path.join(pairs_dir, "left_*.png"))),
                 sorted(glob.glob(os.path.join(pairs_dir, "right_*.png")))))

if not pairs:
    print(f"No image pairs found in {pairs_dir}")
else:
    for L, R in pairs[:5]:
        print(f"Showing {os.path.basename(L)}, {os.path.basename(R)}")
        iL, iR = cv2.imread(L), cv2.imread(R)
        if iL is None or iR is None:
            print("⚠️ Could not read one of the images")
            continue
        overlay = cv2.addWeighted(iL, 0.5, iR, 0.5, 0)
        cv2.imshow("check", overlay)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
