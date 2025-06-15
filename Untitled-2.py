import os
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import uuid
from tensorflow.keras.models import load_model
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()
socketio = SocketIO(app, cors_allowed_origins="*")

# Model will be loaded on demand
model = None

# Define global variables for sign language recognition
actions = np.array(['Hello', 'Thankyou', "I love you"])
sequence_length = 30
threshold = 0.5
hello_reduction_factor = 0.05
i_l_y_incr_fact = 2

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Store active rooms
active_rooms = {}

# Helper functions for sign language recognition
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Load model function
def load_model_on_demand():
    global model
    if model is None:
        try:
            model = load_model('action.h5')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            model = None

# Process frame with sign language detection
def process_frame_with_sign_detection(frame_data, user_id, room_id):
    global model
    try:
        if model is None:
            return None, ""
        
        # Decode base64 image
        encoded_data = frame_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None, ""
        
        # Initialize sequence for this user if not exists
        if user_id not in active_rooms[room_id]['sequences']:
            active_rooms[room_id]['sequences'][user_id] = []
            active_rooms[room_id]['sentences'][user_id] = []
            active_rooms[room_id]['predictions'][user_id] = []
        
        # Get user's sequence data
        sequence = active_rooms[room_id]['sequences'][user_id]
        sentence = active_rooms[room_id]['sentences'][user_id]
        predictions = active_rooms[room_id]['predictions'][user_id]
        
        # Process with MediaPipe
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks if enabled
            if active_rooms[room_id]['show_landmarks']:
                draw_styled_landmarks(image, results)
            
            # Extract keypoints and add to sequence
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]
            
            # Update user's sequence
            active_rooms[room_id]['sequences'][user_id] = sequence
            
            # Make prediction when we have enough frames
            caption_text = ""
            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                res[0] *= hello_reduction_factor
                res[2] *= i_l_y_incr_fact
                res = res / np.sum(res)
                predictions.append(np.argmax(res))
                
                # Update predictions
                active_rooms[room_id]['predictions'][user_id] = predictions
                
                # Check if we have consistent predictions
                if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
                
                # Limit sentence length
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                
                # Update sentence
                active_rooms[room_id]['sentences'][user_id] = sentence
                
                # Add visualization
                colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
                image = prob_viz(res, actions, image, colors)
                
                # Get caption text
                caption_text = ' '.join(sentence)
            
            # Add caption to image
            if caption_text:
                cv2.rectangle(image, (0, 0), (frame.shape[1], 40), (245, 117, 16), -1)
                cv2.putText(image, caption_text, (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Encode processed image
            _, buffer = cv2.imencode('.jpg', image)
            processed_frame = base64.b64encode(buffer).decode('utf-8')
            
            return processed_frame, caption_text
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return None, ""

# Routes
@app.route('/')
def index():
    return render_template('simple_video_call.html')

@app.route('/create-room')
def create_room():
    room_id = str(uuid.uuid4())[:8]
    active_rooms[room_id] = {
        'users': {},
        'sign_language_enabled': False,
        'show_landmarks': False,
        'sequences': {},
        'sentences': {},
        'predictions': {}
    }
    return jsonify({'roomId': room_id})

@app.route('/join-room/<room_id>')
def join_room_page(room_id):
    if room_id in active_rooms:
        return render_template('simple_video_call.html')
    return jsonify({'error': 'Room not found'}), 404

# Socket events
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    for room_id in list(active_rooms.keys()):
        if request.sid in active_rooms[room_id]['users']:
            user_id = active_rooms[room_id]['users'][request.sid]['user_id']
            del active_rooms[room_id]['users'][request.sid]
            if user_id in active_rooms[room_id]['sequences']:
                del active_rooms[room_id]['sequences'][user_id]
            if user_id in active_rooms[room_id]['sentences']:
                del active_rooms[room_id]['sentences'][user_id]
            if user_id in active_rooms[room_id]['predictions']:
                del active_rooms[room_id]['predictions'][user_id]
            emit('user-disconnected', {'userId': user_id}, room=room_id)
            if not active_rooms[room_id]['users']:
                del active_rooms[room_id]

@socketio.on('join-room')
def handle_join_room(data):
    room_id = data['roomId']
    user_id = data['userId']
    
    if room_id not in active_rooms:
        emit('error', {'message': 'Room not found'})
        return
    
    join_room(room_id)
    active_rooms[room_id]['users'][request.sid] = {
        'user_id': user_id,
        'username': data.get('username', f"User-{user_id[:4]}")
    }
    
    emit('user-connected', {
        'userId': user_id,
        'username': active_rooms[room_id]['users'][request.sid]['username']
    }, room=room_id, include_self=False)
    
    existing_users = [{
        'userId': user_data['user_id'],
        'username': user_data['username']
    } for user_data in active_rooms[room_id]['users'].values() if user_data['user_id'] != user_id]
    
    emit('existing-users', {
        'users': existing_users,
        'signLanguageEnabled': active_rooms[room_id]['sign_language_enabled'],
        'showLandmarks': active_rooms[room_id]['show_landmarks']
    })

@socketio.on('video-frame')
def handle_video_frame(data):
    room_id = data['roomId']
    user_id = data['userId']
    frame_data = data['frameData']
    
    if room_id not in active_rooms:
        return
    
    if active_rooms[room_id]['sign_language_enabled']:
        if model is None:
            load_model_on_demand()
        processed_frame, caption = process_frame_with_sign_detection(frame_data, user_id, room_id)
        if processed_frame:
            emit('video-frame', {
                'userId': user_id,
                'frameData': f"data:image/jpeg;base64,{processed_frame}",
                'caption': caption
            }, room=room_id)
    else:
        emit('video-frame', {
            'userId': user_id,
            'frameData': frame_data,
            'caption': ''
        }, room=room_id, include_self=False)

@socketio.on('toggle-sign-language')
def handle_toggle_sign_language(data):
    room_id = data['roomId']
    enabled = data['enabled']
    
    if room_id in active_rooms:
        active_rooms[room_id]['sign_language_enabled'] = enabled
        if enabled and model is None:
            load_model_on_demand()
        emit('sign-language-toggled', {'enabled': enabled}, room=room_id)

@socketio.on('toggle-landmarks')
def handle_toggle_landmarks(data):
    room_id = data['roomId']
    show = data['show']
    
    if room_id in active_rooms:
        active_rooms[room_id]['show_landmarks'] = show
        emit('landmarks-toggled', {'show': show}, room=room_id)

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)