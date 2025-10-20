# check_calibration_quality.py
import os
import numpy as np

def main():
    base_dir = os.path.dirname(__file__)
    calib_path = os.path.join(base_dir, "stereo_calib.npz")

    if not os.path.exists(calib_path):
        print(f"❌ Calibration file not found: {calib_path}")
        return

    data = np.load(calib_path, allow_pickle=True)

    # Extract stored values
    rms = data.get("rms", None)
    errL = data.get("errL", None)
    errR = data.get("errR", None)
    K1 = data.get("K1", None)
    D1 = data.get("D1", None)
    K2 = data.get("K2", None)
    D2 = data.get("D2", None)
    R = data.get("R", None)
    T = data.get("T", None)

    print("\n=== Stereo Calibration Quality Report ===")
    print(f"File: {os.path.basename(calib_path)}\n")

    if rms is not None:
        print(f"RMS reprojection error: {rms:.4f}")
    if errL is not None:
        print(f"Mean reprojection error (Left): {errL:.4f}")
    if errR is not None:
        print(f"Mean reprojection error (Right): {errR:.4f}")

    # Intrinsics summary
    print("\n--- Left Camera ---")
    print("K1:\n", K1)
    print("D1:", D1.ravel())

    print("\n--- Right Camera ---")
    print("K2:\n", K2)
    print("D2:", D2.ravel())

    # Extrinsics summary
    print("\n--- Extrinsics ---")
    print("Rotation (R):\n", R)
    print("Translation (T):\n", T.ravel())

    # Quick quality assessment
    print("\n--- Quality Assessment ---")
    if rms < 1.0 and errL < 1.5 and errR < 1.5:
        print("✅ Excellent calibration quality.")
    elif rms < 2.0 and errL < 3.0 and errR < 3.0:
        print("⚠️ Acceptable but could improve.")
    else:
        print("❌ Poor calibration quality — consider recalibrating.")

if __name__ == "__main__":
    main()
