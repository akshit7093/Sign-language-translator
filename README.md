# Sign Language Video Call Application

A real-time video calling application with integrated sign language detection and translation capabilities. This application enables users to communicate through video calls while providing automatic sign language recognition and live captions.

## 🌟 Features

- **Real-time Video Calling**: WebRTC-based peer-to-peer video communication
- **Sign Language Detection**: AI-powered recognition of sign language gestures
- **Live Captions**: Real-time translation of sign language to text
- **Landmark Visualization**: Optional display of MediaPipe pose/hand landmarks
- **Multi-Camera Support**: Switch between different camera devices
- **Room-based Communication**: Create or join rooms with unique codes
- **Responsive UI**: Modern Bootstrap-based interface

## 🏗️ Architecture

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   AI Models     │
│   (Browser)     │◄──►│   (FastAPI)     │◄──►│   (TensorFlow)  │
│                 │    │                 │    │                 │
│ • WebRTC        │    │ • WebSocket     │    │ • Keras Model   │
│ • JavaScript    │    │ • Room Mgmt     │    │ • MediaPipe     │
│ • HTML/CSS      │    │ • Frame Proc    │    │ • Sign Detect   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow
1. **Video Capture**: Browser captures video frames from user's camera
2. **Frame Processing**: Frames sent to backend for sign language analysis
3. **AI Detection**: MediaPipe extracts landmarks, Keras model predicts signs
4. **Real-time Communication**: WebSocket broadcasts results to all participants
5. **Display**: Captions and landmarks displayed on video feeds

## 📁 Project Structure

```
sign-language-video-call/
├── app_main.py                 # Main FastAPI application
├── sign_language_module.py     # Sign language detection module
├── action.h5                   # Pre-trained Keras model
├── static/
│   ├── script.js              # Frontend JavaScript logic
│   └── style.css              # Styling (optional)
├── templates/
│   └── simple_video_call.html # Main HTML template
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Webcam/Camera device
- Modern web browser (Chrome, Firefox, Safari)

### 1. Clone Repository
```bash
git clone <repository-url>
cd sign-language-video-call
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model
Ensure `action.h5` (pre-trained sign language model) is in the project root directory.

### 5. Run Application
```bash
python app_main.py
```

### 6. Access Application
Open your browser and navigate to: `http://localhost:8000`

## 📋 Dependencies

```txt
fastapi==0.104.1
uvicorn==0.24.0
websockets==12.0
opencv-python==4.8.1.78
mediapipe==0.10.7
tensorflow==2.13.0
numpy==1.24.3
jinja2==3.1.2
python-multipart==0.0.6
```

## 🎯 Core Components

### 1. Backend (`app_main.py`)

**FastAPI Application** serving as the main server:

- **WebSocket Management**: Handles real-time communication
- **Room System**: Manages user sessions and room-based communication
- **Frame Processing**: Coordinates sign language detection
- **User Management**: Tracks participants and their states

**Key Classes:**
- `Room`: Manages participants, detectors, and room state
- `ConnectionManager`: Handles WebSocket connections
- `SignDetector`: Per-user sign language detection instance

### 2. AI Module (`sign_language_module.py`)

**Sign Language Detection Engine**:

- **MediaPipe Integration**: Extracts pose, face, and hand landmarks
- **Keras Model**: Predicts sign language gestures from landmark sequences
- **Real-time Processing**: Processes video frames at 10 FPS
- **State Management**: Maintains detection history per user

**Supported Signs:**
- Hello
- Thank you
- I love you

**Detection Parameters:**
- Sequence Length: 30 frames
- Confidence Threshold: 0.7
- Prediction Consistency: 5 consecutive frames

### 3. Frontend (`static/script.js`)

**Browser-based Client**:

- **WebRTC Implementation**: Peer-to-peer video communication
- **Camera Management**: Multi-camera support and switching
- **Real-time UI**: Live caption display and landmark visualization
- **WebSocket Client**: Communication with backend server

**Key Features:**
- Camera device enumeration and selection
- Frame capture and transmission (10 FPS)
- Real-time caption display
- Landmark overlay rendering

### 4. UI Template (`templates/simple_video_call.html`)

**Responsive Web Interface**:

- Bootstrap-based responsive design
- Video grid layout
- Control panel with toggles
- Room creation/joining interface

## 🚀 Usage Guide

### Starting a Call

1. **Create Room**: Click "Create New Room" to generate a unique room code
2. **Join Room**: Enter an existing room code and click "Join Room"
3. **Grant Permissions**: Allow camera and microphone access when prompted

### Using Sign Language Detection

1. **Enable Detection**: Click the "Sign Language" toggle button
2. **Perform Signs**: Make sign language gestures in front of the camera
3. **View Captions**: Live translations appear below your video feed
4. **Toggle Landmarks**: Enable landmark visualization to see detection points

### Camera Controls

1. **Switch Camera**: Use the camera dropdown to select different devices
2. **Toggle Video/Audio**: Use control buttons to mute/unmute
3. **End Call**: Click the red phone button to leave the room

### Multi-User Features

- **Real-time Sharing**: All participants see each other's captions
- **Independent Controls**: Each user can toggle their own sign language detection
- **Synchronized Display**: Captions and landmarks update in real-time

## ⚙️ Configuration

### Model Parameters

```python
# In sign_language_module.py
class SignDetector:
   def __init__(self, threshold=0.7, sequence_length=30):
       self.threshold = threshold              # Confidence threshold
       self.sequence_length = sequence_length  # Frames for prediction
       self.hello_reduction_factor = 0.05      # Reduce "Hello" sensitivity
       self.i_l_y_incr_fact = 1.5             # Increase "I love you" sensitivity
```

### MediaPipe Settings

```python
holistic_model = mp_holistic.Holistic(
   min_detection_confidence=0.7,    # Detection confidence
   min_tracking_confidence=0.7,     # Tracking confidence
   model_complexity=1,              # Model complexity (0-2)
   refine_face_landmarks=False      # Disable face refinement
)
```

### Frame Processing

```javascript
// In script.js
const frameRate = 10;  // Frames per second for sign detection
const frameQuality = 0.8;  // JPEG quality for frame transmission
```

## 🔍 API Reference

### WebSocket Messages

#### Client to Server

**Toggle Sign Language**
```json
{
   "type": "toggle_sign_language",
   "enable": true
}
```

**Toggle Landmarks**
```json
{
   "type": "toggle_landmarks", 
   "enable": true
}
```

**Video Frame**
```json
{
   "type": "video_frame_for_sign",
   "frame_data": "data:image/jpeg;base64,..."
}
```

**WebRTC Signaling**
```json
{
   "type": "webrtc_signal",
   "sdp": {...},
   "candidate": {...},
   "to": "user_id"
}
```

#### Server to Client

**Caption Update**
```json
{
   "type": "caption_update",
   "user_id": "User-abc123",
   "username": "User-abc123", 
   "caption": "Hello Thank you",
   "current_sign": "Thank you",
   "processed_frame": "data:image/jpeg;base64,..." // Optional
}
```

**User List**
```json
{
   "type": "user_list",
   "users": [
       {"user_id": "User-abc123", "username": "User-abc123"}
   ]
}
```

**Status Updates**
```json
{
   "type": "sign_language_status_update",
   "user_id": "User-abc123",
   "enabled": true
}
```

### HTTP Endpoints

- `GET /`: Main application interface
- `WebSocket /ws/{room_id}/{user_id}`: WebSocket connection endpoint

## 🎨 UI Components

### Video Grid
- **Local Video**: User's own camera feed with self-captions
- **Remote Video**: Other participants' feeds with their captions
- **Responsive Layout**: Adapts to different screen sizes

### Control Panel
- **Video Toggle**: Enable/disable camera
- **Audio Toggle**: Mute/unmute microphone  
- **Sign Language Toggle**: Enable/disable sign detection
- **Landmarks Toggle**: Show/hide pose landmarks
- **Camera Selector**: Switch between available cameras
- **End Call**: Leave the room

### Caption Display
- **Real-time Updates**: Live sign language translations
- **User Identification**: Color-coded borders (blue=local, green=remote)
- **Status Indicators**: Shows "Listening for signs..." when active
- **Overlay Design**: Semi-transparent overlay on video feeds

## 🔧 Troubleshooting

### Common Issues

**Model Loading Errors**
```
Error: Model file not found at action.h5
```
**Solution**: Ensure `action.h5` is in the project root directory

**Camera Access Denied**
```
Error: Could not access camera or microphone
```
**Solution**: Grant camera/microphone permissions in browser settings

**WebSocket Connection Failed**
```
WebSocket connection error
```
**Solution**: Check if server is running on correct port (8000)

**Sign Detection Not Working**
```
ValueError: Input shape mismatch
```
**Solution**: Verify model expects 30-frame sequences with 1662 features

### Performance Optimization

**Reduce Lag**:
- Lower frame rate in `script.js` (frameRate = 5)
- Disable face landmarks in MediaPipe
- Reduce video resolution

**Improve Accuracy**:
- Increase confidence threshold (0.8+)
- Ensure good lighting conditions
- Position hands clearly in camera view

### Debug Mode

Enable debug logging:
```python
# In sign_language_module.py
print(f"Frame {self.frame_count}: Current='{current_sign}', Sentence='{sentence_text}'")
```

Browser console commands:
```javascript
// List available cameras
listCameras();

// Switch to specific camera
switchToCamera(1);

// Get current camera info
getCurrentCameraInfo();
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe**: Google's framework for building perception pipelines
- **TensorFlow**: Machine learning platform for model training and inference
- **FastAPI**: Modern, fast web framework for building APIs
- **WebRTC**: Real-time communication protocol for peer-to-peer connections

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review browser console for error messages
3. Verify all dependencies are installed correctly
4. Ensure camera permissions are granted

---

**Note**: This application requires a pre-trained sign language model (`action.h5`) trained on the specific signs mentioned above. The model architecture expects sequences of 30 frames with 1662 features extracted from MediaPipe landmarks.