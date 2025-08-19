import cv2

def test_laptop_camera():
    """Test laptop camera and show feed"""
    print("Testing laptop camera...")
    
    # Try different camera indices
    for camera_index in range(5):
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera {camera_index} is working!")
                print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
                
                # Show camera feed
                print("Showing camera feed... Press 'q' to try next camera or ESC to exit")
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Flip frame horizontally (mirror effect)
                    frame = cv2.flip(frame, 1)
                    
                    # Add text
                    cv2.putText(frame, f"Camera Index: {camera_index}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Press 'q' for next camera, ESC to exit", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow(f"Camera Test - Index {camera_index}", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        cap.release()
                        cv2.destroyAllWindows()
                        print("Test completed!")
                        return camera_index
                    elif key == ord('q'):  # 'q' key
                        break
                
                cap.release()
                cv2.destroyWindow(f"Camera Test - Index {camera_index}")
            else:
                print(f"✗ Camera {camera_index} opened but cannot read frames")
                cap.release()
        else:
            print(f"✗ Camera {camera_index} cannot be opened")
    
    print("No working camera found!")
    return -1

if __name__ == "__main__":
    working_camera = test_laptop_camera()
    if working_camera >= 0:
        print(f"\nRecommendation: Use camera index {working_camera} in your virtual mouse code")
        print(f"Change this line: cap = cv2.VideoCapture({working_camera})")
    else:
        print("\nNo camera detected. Please check:")
        print("1. Camera is not being used by another app (Zoom, Skype, etc.)")
        print("2. Camera drivers are properly installed")
        print("3. Privacy settings allow camera access")