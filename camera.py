import cv2

def open_camera():
    # Initialize camera (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if ret:
            # Display the frame
            cv2.imshow('Camera Feed', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Camera feed ended")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_camera()
