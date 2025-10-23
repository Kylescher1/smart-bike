import numpy as np, json
data = np.load("depth_20251023_145210.npz")
disp = data["disp"]
num_disp = int(data["num_disp"])
settings = json.loads(data["settings"].item())
