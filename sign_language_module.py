import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
import traceback

# --- MediaPipe and Model Initialization ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load the pre-trained Keras model for sign language detection
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'action.h5')
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        # Print model input shape for debugging
        if hasattr(model, 'input_shape'):
            print(f"Model expects input shape: {model.input_shape}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print("Sign language detection will not work without the model file.")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()

# Initialize MediaPipe Holistic model instance with proper configuration
holistic_model = None
try:
    # Use higher confidence thresholds and proper configuration
    holistic_model = mp_holistic.Holistic(
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7,
        model_complexity=1,  # Use lighter model for better performance
        refine_face_landmarks=False  # Disable face refinement to reduce warnings
    )
    print("MediaPipe Holistic model initialized successfully")
except Exception as e:
    print(f"Error initializing MediaPipe: {e}")

# Define the actions (signs) that the model can recognize
ACTIONS = np.array(['Hello', 'Thankyou', 'I love you'])

# --- Core Detection and Processing Functions ---
def mediapipe_detection(image, holistic_pipe):
    """Processes an image using MediaPipe Holistic model to detect landmarks."""
    try:
        if holistic_pipe is None:
            return None
        
        # Ensure proper image dimensions for MediaPipe
        height, width = image.shape[:2]
        if width > 640:
            image = cv2.resize(image, (640, 480))
        
        # Ensure image is in correct format
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        image_rgb.flags.writeable = False
        results = holistic_pipe.process(image_rgb)
        image_rgb.flags.writeable = True
        return results
    except Exception as e:
        print(f"Error in mediapipe_detection: {e}")
        return None

def extract_keypoints(results):
    """Extracts keypoints from MediaPipe results into a flat NumPy array."""
    try:
        if results is None:
            return np.zeros(1662)  # Total expected keypoints
            
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, face, lh, rh])
    except Exception as e:
        print(f"Error in extract_keypoints: {e}")
        return np.zeros(1662)

# --- Sign Detector Class for Per-User State Management ---
# SIGN ADJUSTMENTS - Modified to favor Hello over I love you
SIGN_ADJUSTMENTS = {
    # Increase sensitivity for Hello
    'Hello': 0.6,           # Significantly boost Hello detection
    'Thankyou': 1.0,        # Keep Thank you at normal level
    'I love you': 1.0,      # Reduce I love you sensitivity
    
    # Default for any other signs
    'default': 1.0
}

# --- Sign Detector Class - Modified process_frame method ---
class SignDetector:
    def __init__(self, threshold=0.7, sequence_length=30):
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.threshold = threshold
        self.sequence_length = sequence_length
        # Modified factors to favor Hello
        self.hello_boost_factor = 2.5      # Boost Hello significantly
        self.i_l_y_reduction_factor = 0.3  # Reduce "I love you" significantly
        self.frame_count = 0
        self.last_prediction = None
        self.prediction_count = 0
        print(f"SignDetector initialized with Hello boost: {self.hello_boost_factor}, I love you reduction: {self.i_l_y_reduction_factor}")

    def process_frame(self, frame):
        """Processes a single video frame for sign language detection."""
        try:
            self.frame_count += 1
            
            if holistic_model is None:
                return "", "", None
                
            # 1. Detect landmarks using the global holistic_model
            results = mediapipe_detection(frame, holistic_model)
            if results is None:
                return "", "", None

            # 2. Extract keypoints
            keypoints = extract_keypoints(results)
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-self.sequence_length:]  # Keep exactly 30 frames

            current_sign = ""

            # Only predict when we have exactly the required sequence length
            if len(self.sequence) == self.sequence_length and model is not None:
                try:
                    # 3. Predict using the global Keras model
                    sequence_array = np.expand_dims(self.sequence, axis=0)
                    print(f"Prediction input shape: {sequence_array.shape}")  # Debug print
                    
                    res = model.predict(sequence_array, verbose=0)[0]
                    
                    # 4. Apply custom adjustments - MODIFIED TO FAVOR HELLO
                    # res[0] *= self.hello_boost_factor      # Significantly boost Hello (index 0)
                    # res[2] *= self.i_l_y_reduction_factor  # Significantly reduce "I love you" (index 2)
                    # res[1] remains unchanged for "Thank you"
                    
                    # 5. Normalize probabilities
                    if np.sum(res) > 0:
                        res = res / np.sum(res)
                    
                    predicted_idx = np.argmax(res)
                    max_prob = res[predicted_idx]
                    
                    # 6. Additional Hello preference logic
                    hello_prob = res[0]
                    i_love_you_prob = res[2]
                    
                    # If Hello probability is reasonably close to the max, prefer Hello
                    if hello_prob > 0.9 and predicted_idx == 2:  # If "I love you" was predicted
                        if hello_prob / i_love_you_prob > 0.9:    # If Hello is at least 60% of I love you
                            predicted_idx = 0  # Switch to Hello
                            max_prob = hello_prob
                            print(f"Switched prediction from 'I love you' to 'Hello' - Hello: {hello_prob:.3f}, I love you: {i_love_you_prob:.3f}")
                    
                    # 7. Confidence check
                    if max_prob > self.threshold:
                        self.predictions.append(predicted_idx)
                        
                        # More stringent prediction verification (from import cv2 copy.py)
                        if len(self.predictions) >= 5 and np.unique(self.predictions[-5:])[0] == predicted_idx:
                            predicted_action = ACTIONS[predicted_idx]
                            current_sign = predicted_action
                            
                            # Add to sentence if it's different from the last word
                            if len(self.sentence) > 0:
                                if predicted_action != self.sentence[-1]:
                                    self.sentence.append(predicted_action)
                                    print(f"Added '{predicted_action}' to sentence. Probability: {max_prob:.3f}")
                            else:
                                self.sentence.append(predicted_action)
                                print(f"Added '{predicted_action}' to sentence. Probability: {max_prob:.3f}")
                            
                            # Keep sentence length manageable (from import cv2 copy.py)
                            if len(self.sentence) > 3:  # Reduced sentence length for faster updates
                                self.sentence = self.sentence[-3:]
                
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    traceback.print_exc()
            
            sentence_text = ' '.join(self.sentence)
            
            # Debug output every 60 frames (reduced frequency)
            if self.frame_count % 60 == 0:
                print(f"Frame {self.frame_count}: Sequence length: {len(self.sequence)}, Current='{current_sign}', Sentence='{sentence_text}'")
            
            return current_sign, sentence_text, results
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            traceback.print_exc()
            return "", "", None

    def reset(self):
        """Reset the detector state."""
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.frame_count = 0
        self.last_prediction = None
        self.prediction_count = 0
        print("SignDetector state reset")

# Optional: Function to draw landmarks and probabilities
def draw_styled_landmarks(image, results):
    """Draw styled landmarks on the image - optimized version."""
    try:
        if results is None:
            return
        
        # Create a transparent overlay for landmarks
        overlay = image.copy()
        
        # Only draw essential landmarks to reduce lag
        
        # Pose (reduced thickness for performance)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                overlay, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )
        
        # Left Hand
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                overlay, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
        
        # Right Hand
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                overlay, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        # Blend the overlay with the original image
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
    except Exception as e:
        print(f"Error drawing landmarks: {e}")

def prob_viz(res, actions, input_frame, colors):
    """Visualize prediction probabilities on the frame."""
    try:
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            if num < len(actions) and num < len(colors):
                cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
                cv2.putText(output_frame, f"{actions[num]}: {prob:.2f}", (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return output_frame
    except Exception as e:
        print(f"Error in prob_viz: {e}")
        return input_frame

# Colors for probability visualization
PROB_COLORS = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# Cleanup function
def cleanup():
    """Clean up MediaPipe resources."""
    global holistic_model
    if holistic_model:
        holistic_model.close()
        holistic_model = None
        print("MediaPipe resources cleaned up")
