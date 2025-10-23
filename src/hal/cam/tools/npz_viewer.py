import argparse
from pathlib import Path
import numpy as np, json, sys

def find_latest_npz(folder: Path):
    files = sorted(folder.glob('*.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

parser = argparse.ArgumentParser(description='View .npz depth file')
parser.add_argument('file', nargs='?', help='path to .npz file (optional). If omitted the newest file in C:\\smart-bike\\images will be used.')
args = parser.parse_args()

if args.file:
    fp = Path(args.file)
else:
    images_dir = Path(r'C:\smart-bike\images')
    fp = find_latest_npz(images_dir)

if not fp or not fp.exists() or fp.is_dir():
    print("Error: .npz file not found.")
    images_dir = Path(r'C:\smart-bike\images')
    if images_dir.exists():
        print("Candidate files in C:\\smart-bike\\images:")
        for p in images_dir.glob('*.npz'):
            print("  ", p.name)
    sys.exit(1)

data = np.load(str(fp), allow_pickle=True)
disp = data.get("disp")
num_disp = int(data.get("num_disp", 0)) if "num_disp" in data else None
settings = None
if "settings" in data:
    try:
        settings = json.loads(data["settings"].item())
    except Exception:
        settings = data["settings"].item()

print("Loaded:", fp)