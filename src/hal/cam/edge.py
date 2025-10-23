import cv2
import numpy as np

# Load grayscale image
img = cv2.imread("right_015.PNG", cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (5, 5), 1.4)
edges = cv2.Canny(blur, 80, 160)

# Close small gaps in edges
kernel = np.ones((3, 3), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

# Invert edges to prepare for connected components
inv = cv2.bitwise_not(closed)

# Label connected regions
num_labels, labels = cv2.connectedComponents(inv)

# Map each label to color
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
segmented = cv2.merge([label_hue, blank_ch, blank_ch])
segmented = cv2.cvtColor(segmented, cv2.COLOR_HSV2BGR)
segmented[label_hue == 0] = 0

cv2.imshow("Edges", edges)
cv2.imshow("Segmented", segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()
