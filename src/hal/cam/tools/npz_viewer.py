import argparse
from pathlib import Path
import numpy as np, json, sys, cv2

def find_latest_npz(folder: Path):
    files = sorted(folder.glob('*.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

parser = argparse.ArgumentParser(description='View .npz depth file')
parser.add_argument('file', nargs='?', help='path to .npz file (optional). If omitted the newest file in C:\\smart-bike\\images will be used.')
args = parser.parse_args()

# Choose file
if args.file:
    fp = Path(args.file)
else:
    images_dir = Path(r'C:\smart-bike\images')
    fp = find_latest_npz(images_dir)

# Validate
if not fp or not fp.exists() or fp.is_dir():
    print("Error: .npz file not found.")
    images_dir = Path(r'C:\smart-bike\images')
    if images_dir.exists():
        print("Candidate files in C:\\smart-bike\\images:")
        for p in images_dir.glob('*.npz'):
            print("  ", p.name)
    sys.exit(1)

# Load contents
data = np.load(str(fp), allow_pickle=True)
disp = data.get("disp")
num_disp = int(data.get("num_disp", 0)) if "num_disp" in data else None

# Parse settings JSON
settings = None
if "settings" in data:
    try:
        settings = json.loads(data["settings"].item())
    except Exception:
        settings = data["settings"].item()

# Summary
print(f"\nLoaded file: {fp}")
print(f"Disparity map shape: {disp.shape if disp is not None else 'None'}")
print(f"Disparity type: {disp.dtype if disp is not None else 'N/A'}")
if disp is not None:
    print(f"Disparity range: {disp.min():.2f} .. {disp.max():.2f}")
print(f"num_disp: {num_disp}")
print("\nSettings:")
print(json.dumps(settings, indent=2) if settings else "(none)")

# Visualization
if disp is not None:
    # Normalize disparity for viewing
    valid = disp > 0
    if np.any(valid):
        dmin, dmax = np.percentile(disp[valid], [2, 98])
    else:
        dmin, dmax = 0, 1
    disp_norm = np.clip((disp - dmin) / (dmax - dmin), 0, 1)
    disp_color = cv2.applyColorMap((disp_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imshow("Disparity Map", disp_color)
    print("\nPress any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
