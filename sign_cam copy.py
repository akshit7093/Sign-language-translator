import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyvirtualcam

# Load model and define global variables
try:
    model = load_model('action.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

sequence = []
sentence = []
predictions = []
threshold = 0.7
actions = np.array(['Hello', 'Thankyou', 'I love you'])
no_sequence = 30  # Thirty videos worth of data
sequence_length = 30  # Videos are going to be 30 frames in length

# Mediapipe holistic model setup
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color conversion
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Color conversion
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (10, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def main():
    hello_reduction_factor = 0.8

    global sequence, sentence, predictions
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Get the width and height of the frame dynamically
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        # Create a virtual camera with the dynamic width and height
        sign_cam = pyvirtualcam.Camera(width=width, height=height, fps=30, print_fps=True)
    except Exception as e:
        print(f"Error initializing virtual camera: {e}")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # res[0] *= hello_reduction_factor
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

                # Ensure the captions remain inside the frame
                if cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0][0] > width:
                    sentence = [actions[np.argmax(res)]]

                # Add black box behind the captions
                box_height = 70  # Increased box height to accommodate larger text
                cv2.rectangle(image, (0, image.shape[0] - box_height - 50), (image.shape[1], image.shape[0] - 50), (0, 0, 0), -1)

                # Add predicted word as caption at the bottom of the frame, moved up and increased size
                cv2.putText(image, ' '.join(sentence), (10, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Flip the image horizontally just before sending to virtual camera
            # flipped_image = cv2.flip(image, 1)
            
            # Send the flipped frame to the virtual camera
            sign_cam.send(image)
            sign_cam.sleep_until_next_frame()

    cap.release()
    cv2.destroyAllWindows()
    sign_cam.close()

if __name__ == '__main__':
    main()