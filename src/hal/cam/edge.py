import cv2

# Load image in grayscale
img = cv2.imread("right_015.PNG", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found.")

# Create window and trackbars for thresholds
cv2.namedWindow("Edges")
cv2.createTrackbar("Low", "Edges", 50, 255, lambda x: None)
cv2.createTrackbar("High", "Edges", 150, 255, lambda x: None)

while True:
    low = cv2.getTrackbarPos("Low", "Edges")
    high = cv2.getTrackbarPos("High", "Edges")
    blur = cv2.GaussianBlur(img, (5, 5), 1.4)
    edges = cv2.Canny(blur, low, high)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
