import cv2
import sys

def test_webcam(camera_index=0):
    """
    Simple webcam testing function
    
    :param camera_index: Camera device index (default 0)
    """
    # Open the camera
    cap = cv2.VideoCapture(camera_index)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return False
    
    print(f"Successfully opened camera {camera_index}")
    
    # Get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    
    # Capture and display frames
    try:
        frame_count = 0
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # If frame is read correctly ret is True
            if not ret:
                print("Failed to grab frame")
                break
            
            # Display the resulting frame
            cv2.imshow(f'Webcam Test (Camera {camera_index})', frame)
            
            # Increment frame counter
            frame_count += 1
            
            # Break the loop after showing 300 frames or on 'q' key press
            if frame_count > 300 or cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"Captured {frame_count} frames")
    return True

def main():
    # Try multiple camera indices
    camera_indices = [0, 1, 2]  # Most common webcam indices
    
    for index in camera_indices:
        print(f"\nTesting camera index: {index}")
        if test_webcam(index):
            break

if __name__ == "__main__":
    main()
