import numpy as np

data = np.load("stereo_calib_fisheye.npz")
print("K1:\n", data["K1"])
print("K2:\n", data["K2"])
print("R:\n", data["R"])
print("T:\n", data["T"])
