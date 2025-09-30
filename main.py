from src.hal.cam.Camera import open_stereo_pair
from src.hal.cam.depth import get_static_depth, disparity_to_points
from src.hal.cam.calib import load_calibration

import cv2, time, numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def main():
    time.sleep(15)
    calib = load_calibration()
    disp, vis = get_static_depth(show=False)
    if disp is None:
        return
    points = disparity_to_points(disp, calib)

    # mouse callback
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            X, Y, Z = points[y, x]
            Z = Z * 0.03937
            print(f"Pixel ({x},{y}) -> X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}")

            # optional: overlay text directly at click
            cv2.putText(vis, f"{Z:.1f}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow("Depth Map", vis)

    cv2.imshow("Depth Map", vis)
    cv2.setMouseCallback("Depth Map", on_click)

    print("Click on the depth map to see 3D coordinates. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mask = np.isfinite(points[:, :, 2]) & (points[:, :, 2] > 0)
    pts = points[mask]
    cols = vis[mask] / 255.0

    # Always build the Open3D object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    print(f"Point cloud has {len(pts)} points")

    print("Falling back to Matplotlib scatter plot...")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # downsample for speed
    sample = pts[::500]
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1)
    plt.show()


if __name__ == "__main__":
    main()
