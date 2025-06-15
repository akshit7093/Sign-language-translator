from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import mediapipe as mp
from tensorflow.keras.initializers import glorot_uniform

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import glorot_uniform

# Load the trained model
model = load_model('action.h5', custom_objects={'glorot_uniform': glorot_uniform})


# Define actions
actions = ['hello', 'thanks', 'iloveyou']
def extract_keypoints(results):
    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.append((landmark.x, landmark.y, landmark.z))
    return np.array(keypoints).flatten()

# Function to generate video frames
def generate_frames():
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                # Perform necessary preprocessing on the frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Make predictions using the model
                results = holistic.process(image)

                # Display predictions on the frame
                if results.pose_landmarks:
                    # Get keypoints and preprocess
                    keypoints = extract_keypoints(results)
                    keypoints = np.expand_dims(keypoints, axis=0)
                    # Make predictions
                    predictions = model.predict(keypoints)
                    action = actions[np.argmax(predictions)]
                    # Draw text on the frame
                    cv2.putText(frame, action, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Convert the frame to JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # Yield the frame in byte format
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route to stream video frames"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
