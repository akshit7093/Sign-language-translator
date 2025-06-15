import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyvirtualcam
import time
from tensorflow.keras.layers import LSTM

# Custom LSTM to handle time_major
class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        # Remove time_major if it's there
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

# Use custom_objects to override the default LSTM with CustomLSTM
model = load_model('action.h5', custom_objects={'LSTM': CustomLSTM})

sequence = []
sentence = []
predictions = []
threshold = 0.7
actions = np.array(['Hello', 'Thankyou', 'I love you'])
sequence_length = 30

# Mediapipe holistic model setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    height, width, _ = image.shape
    image.flags.writeable = False
    results = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image.flags.writeable = True
    return image, results

def draw_styled_landmarks(image, results):
    # Drawing specifications
    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
    connection_drawing_spec = mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)

    # Draw landmarks
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                              landmark_drawing_spec, connection_drawing_spec)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec, connection_drawing_spec)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec, connection_drawing_spec)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec, connection_drawing_spec)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, f"{actions[num]}: {prob:.2f}", (10, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def main():
    global sequence, sentence, predictions
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set higher FPS for the camera
    cap.set(cv2.CAP_PROP_FPS, 60)

    sign_cam = pyvirtualcam.Camera(width=width, height=height, fps=60)

    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]

            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                image = prob_viz(res, actions, image, colors)

            # UI Enhancements
            # Add a semi-transparent overlay
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
            image = cv2.addWeighted(overlay, 0.1, image, 0.9, 0)

            # Add a stylish header
            cv2.rectangle(image, (0, 0), (width, 50), (245, 117, 16), -1)
            cv2.putText(image, "Sign Language Recognition", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display the recognized sentence (subtitles)
            sentence_text = ' '.join(sentence)
            text_size, _ = cv2.getTextSize(sentence_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
            text_x = (width - text_size[0]) // 2
            
            # Improved subtitle background
            subtitle_height = 80
            cv2.rectangle(image, (0, height - subtitle_height), (width, height), (0, 0, 0), -1)
            cv2.rectangle(image, (0, height - subtitle_height), (width, height), (245, 117, 16), 3)  # Orange border
            
            # Render subtitle text
            cv2.putText(image, sentence_text, (text_x, height - subtitle_height // 2 + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Calculate and display FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(image, f"FPS: {fps:.2f}", (width - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            sign_cam.send(image)
            sign_cam.sleep_until_next_frame()

    cap.release()
    cv2.destroyAllWindows()
    sign_cam.close()

if __name__ == '__main__':
    main()