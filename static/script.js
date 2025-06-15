// DOM elements
const joinForm = document.getElementById('joinForm');
const roomInfo = document.getElementById('roomInfo');
const videoGrid = document.getElementById('videoGrid');
const controls = document.getElementById('controls');
const localVideo = document.getElementById('localVideo');
const localCaption = document.getElementById('localCaption');
const roomIdDisplay = document.getElementById('roomIdDisplay');
const createRoomBtn = document.getElementById('createRoomBtn');
const joinRoomBtn = document.getElementById('joinRoomBtn');
const roomIdInput = document.getElementById('roomIdInput');
const usernameInput = document.getElementById('usernameInput');
const toggleVideoBtn = document.getElementById('toggleVideoBtn');
const toggleAudioBtn = document.getElementById('toggleAudioBtn');
const toggleSignLanguageBtn = document.getElementById('toggleSignLanguageBtn');
const toggleLandmarksBtn = document.getElementById('toggleLandmarksBtn');
const endCallBtn = document.getElementById('endCallBtn');

// Global variables
let socket = null;
let roomId = null;
let username = '';
let localStream = null;
let peers = {};
let videoEnabled = true;
let audioEnabled = true;
let signLanguageEnabled = false;
let showLandmarks = false;
let frameInterval = null;
const frameRate = 8; // Reduced frame rate for better performance (from import cv2 copy.py inspiration)

// Add these variables at the top with other global variables
let availableCameras = [];
let selectedCameraId = null;

// Add camera selection elements
const cameraSelect = document.getElementById('cameraSelect');
const cameraSelectRuntime = document.getElementById('cameraSelectRuntime');

// Initialize
function init() {
    createRoomBtn.addEventListener('click', createRoom);
    joinRoomBtn.addEventListener('click', joinRoom);
    toggleVideoBtn.addEventListener('click', toggleVideo);
    toggleAudioBtn.addEventListener('click', toggleAudio);
    toggleSignLanguageBtn.addEventListener('click', toggleSignLanguage);
    toggleLandmarksBtn.addEventListener('click', toggleLandmarks);
    endCallBtn.addEventListener('click', endCall);
    
    // Add camera selection event listeners
    if (cameraSelect) {
        cameraSelect.addEventListener('change', handleCameraSelection);
    }
    if (cameraSelectRuntime) {
        cameraSelectRuntime.addEventListener('change', handleRuntimeCameraChange);
    }
    
    // Load available cameras on page load
    loadAvailableCameras();
}

// Setup WebSocket event listeners
function setupSocketListeners() {
    if (socket) {
        socket.onopen = handleSocketOpen;
        socket.onmessage = handleSocketMessage;
        socket.onclose = handleSocketClose;
        socket.onerror = handleSocketError;
    }
}

// Handle WebSocket open
function handleSocketOpen(event) {
    console.log('WebSocket connected successfully');
    initializeWebRTC();
}

// Handle incoming WebSocket messages
function handleSocketMessage(event) {
    const data = JSON.parse(event.data);
    console.log('WebSocket message received:', data.type);

    switch (data.type) {
        case 'webrtc_signal':
            handleWebRTCSignal(data);
            break;
        case 'user_list':
            handleUserList(data);
            break;
        case 'caption_update':
            handleCaptionUpdate(data);
            break;
        case 'sign_language_status_update':
            handleSignLanguageStatusUpdate(data);
            break;
        case 'landmarks_status_update':
            handleLandmarksStatusUpdate(data);
            break;
        default:
            console.warn('Unknown message type:', data.type);
    }
}

// Handle WebSocket close
function handleSocketClose(event) {
    console.log('WebSocket closed:', event);
}

// Handle WebSocket error
function handleSocketError(error) {
    console.error('WebSocket error:', error);
    alert('WebSocket connection error. Please try again.');
}

// Generate random ID
function generateId() {
    return Math.random().toString(36).substring(2, 10);
}

// Connect to WebSocket
function connectWebSocket(id, user) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/${id}/${user}`;
    console.log('Attempting to connect to WebSocket:', wsUrl);
    
    roomId = id;
    username = user;
    
    socket = new WebSocket(wsUrl);
    setupSocketListeners();
}

// Create a new room
function createRoom() {
    const user_id = `User-${generateId()}`;
    const tempRoomId = `room_${generateId()}`;
    connectWebSocket(tempRoomId, user_id);
}

// Join an existing room
function joinRoom() {
    const user_id = `User-${generateId()}`;
    const inputRoomId = roomIdInput.value.trim();
    if (!inputRoomId) {
        alert('Please enter a room ID');
        return;
    }
    connectWebSocket(inputRoomId, user_id);
}

// Show room UI
function showRoomUI() {
    joinForm.classList.add('hidden');
    roomInfo.classList.remove('hidden');
    videoGrid.classList.remove('hidden');
    controls.classList.remove('hidden');
    roomIdDisplay.textContent = roomId;
}

// Function to get available camera devices
async function loadAvailableCameras() {
    try {
        // Request permission first
        await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        
        // Get all devices
        const devices = await navigator.mediaDevices.enumerateDevices();
        availableCameras = devices.filter(device => device.kind === 'videoinput');
        
        console.log('Available cameras:', availableCameras);
        
        // Populate camera selection dropdowns
        populateCameraSelectors();
        
    } catch (error) {
        console.error('Error getting camera devices:', error);
    }
}

// Populate camera selector dropdowns
function populateCameraSelectors() {
    const selectors = [cameraSelect, cameraSelectRuntime].filter(Boolean);
    
    selectors.forEach(selector => {
        // Clear existing options except default
        selector.innerHTML = '<option value="">Default Camera</option>';
        
        // Add camera options
        availableCameras.forEach((camera, index) => {
            const option = document.createElement('option');
            option.value = camera.deviceId;
            option.textContent = camera.label || `Camera ${index + 1}`;
            selector.appendChild(option);
        });
    });
}

// Handle camera selection before joining
function handleCameraSelection(event) {
    selectedCameraId = event.target.value;
    console.log('Selected camera:', selectedCameraId);
}

// Handle runtime camera change (during call)
async function handleRuntimeCameraChange(event) {
    const newCameraId = event.target.value;
    
    if (localStream) {
        try {
            // Stop current video track
            const videoTrack = localStream.getVideoTracks()[0];
            if (videoTrack) {
                videoTrack.stop();
            }
            
            // Get new camera stream
            const constraints = {
                video: { 
                    width: 640, 
                    height: 480,
                    deviceId: newCameraId ? { exact: newCameraId } : undefined
                },
                audio: false // Don't change audio
            };
            
            const newVideoStream = await navigator.mediaDevices.getUserMedia(constraints);
            const newVideoTrack = newVideoStream.getVideoTracks()[0];
            
            // Replace video track in peer connection
            if (peerConnection) {
                const sender = peerConnection.getSenders().find(s => 
                    s.track && s.track.kind === 'video'
                );
                if (sender) {
                    await sender.replaceTrack(newVideoTrack);
                }
            }
            
            // Update local stream
            localStream.removeTrack(videoTrack);
            localStream.addTrack(newVideoTrack);
            
            // Update local video element
            localVideo.srcObject = localStream;
            
            selectedCameraId = newCameraId;
            console.log('Camera switched to:', newCameraId);
            
        } catch (error) {
            console.error('Error switching camera:', error);
            alert('Failed to switch camera. Please try again.');
        }
    }
}

// Initialize WebRTC
async function initializeWebRTC() {
    try {
        const constraints = {
            video: { 
                width: 640, 
                height: 480,
                deviceId: selectedCameraId ? { exact: selectedCameraId } : undefined
            },
            audio: true
        };
        
        localStream = await navigator.mediaDevices.getUserMedia(constraints);
        localVideo.srcObject = localStream;
        
        showRoomUI();
        initializePeerConnection();
        createLocalVideoContainer();

        if (signLanguageEnabled) {
            startFrameCapture();
        }
        
        console.log('Using camera:', selectedCameraId || 'default');
        
    } catch (error) {
        console.error('Error accessing media devices:', error);
        
        // Fallback to default camera if selected camera fails
        if (selectedCameraId) {
            console.log('Falling back to default camera...');
            selectedCameraId = null;
            return initializeWebRTC();
        }
        
        alert('Could not access camera or microphone. Please check permissions.');
    }
}

// Function to get camera info (useful for debugging)
function getCurrentCameraInfo() {
    if (localStream) {
        const videoTrack = localStream.getVideoTracks()[0];
        if (videoTrack) {
            const settings = videoTrack.getSettings();
            console.log('Current camera settings:', settings);
            return settings;
        }
    }
    return null;
}

// Function to manually switch to specific camera index
async function switchToCameraIndex(index) {
    if (index < availableCameras.length) {
        const cameraId = availableCameras[index].deviceId;
        
        // Update dropdown
        if (cameraSelectRuntime) {
            cameraSelectRuntime.value = cameraId;
        }
        
        // Trigger camera change
        await handleRuntimeCameraChange({ target: { value: cameraId } });
    }
}

// Add these utility functions for easier camera switching
window.switchToCamera = switchToCameraIndex; // For console debugging
window.listCameras = () => {
    console.log('Available cameras:');
    availableCameras.forEach((camera, index) => {
        console.log(`${index}: ${camera.label || 'Unknown'} (${camera.deviceId})`);
    });
};

// Create local video container with caption
function createLocalVideoContainer() {
    const localVideoContainer = localVideo.parentElement;
    
    // Add caption display for local video
    let localCaptionDisplay = document.getElementById('localCaptionDisplay');
    if (!localCaptionDisplay) {
        localCaptionDisplay = document.createElement('div');
        localCaptionDisplay.id = 'localCaptionDisplay';
        localCaptionDisplay.className = 'caption-display';
        localCaptionDisplay.style.cssText = `
            position: absolute;
            bottom: 10px;
            left: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 14px;
            text-align: center;
            min-height: 25px;
            font-weight: bold;
            display: none;
            z-index: 10;
            border: 2px solid #007bff;
        `;
        localVideoContainer.style.position = 'relative';
        localVideoContainer.appendChild(localCaptionDisplay);
    }
}

let peerConnection;

function initializePeerConnection() {
    const configuration = {
        iceServers: [{
            urls: 'stun:stun.l.google.com:19302'
        }]
    };
    peerConnection = new RTCPeerConnection(configuration);

    peerConnection.onicecandidate = (event) => {
        if (event.candidate && socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
                type: 'webrtc_signal',
                sdp: null,
                candidate: event.candidate
            }));
        }
    };

    peerConnection.ontrack = (event) => {
        console.log('Remote stream received:', event.streams[0]);
        addRemoteStream(event.streams[0], 'Remote User');
    };

    if (localStream) {
        localStream.getTracks().forEach(track => {
            peerConnection.addTrack(track, localStream);
        });
    }
}

function handleWebRTCSignal(data) {
    if (!peerConnection) {
        initializePeerConnection();
    }

    if (data.sdp) {
        if (data.sdp.type === 'offer') {
            peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp))
                .then(() => peerConnection.createAnswer())
                .then(answer => peerConnection.setLocalDescription(answer))
                .then(() => {
                    if (socket && socket.readyState === WebSocket.OPEN) {
                        socket.send(JSON.stringify({
                            type: 'webrtc_signal',
                            sdp: peerConnection.localDescription,
                            candidate: null
                        }));
                    }
                })
                .catch(error => console.error('Error handling offer:', error));
        } else if (data.sdp.type === 'answer') {
            peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp))
                .catch(error => console.error('Error handling answer:', error));
        }
    } else if (data.candidate) {
        peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate))
            .catch(error => console.error('Error adding ICE candidate:', error));
    }
}

function handleUserList(data) {
    console.log('User list received:', data.users);
    if (peerConnection && data.users.length > 1) {
        peerConnection.createOffer()
            .then(offer => peerConnection.setLocalDescription(offer))
            .then(() => {
                if (socket && socket.readyState === WebSocket.OPEN) {
                    socket.send(JSON.stringify({
                        type: 'webrtc_signal',
                        sdp: peerConnection.localDescription,
                        candidate: null
                    }));
                }
            })
            .catch(error => console.error('Error creating offer:', error));
    }
}

function addRemoteStream(stream, username) {
    let remoteVideoContainer = document.getElementById('remoteVideoContainer');
    if (!remoteVideoContainer) {
        const videoContainer = document.createElement('div');
        videoContainer.id = 'remoteVideoContainer';
        videoContainer.className = 'video-container';
        videoContainer.style.position = 'relative';

        const remoteVideo = document.createElement('video');
        remoteVideo.id = 'remoteVideo';
        remoteVideo.autoplay = true;
        remoteVideo.playsInline = true;
        remoteVideo.srcObject = stream;

        const captionElement = document.createElement('div');
        captionElement.className = 'user-name';
        captionElement.textContent = username;

        // Add caption display for remote video
        const remoteCaptionDisplay = document.createElement('div');
        remoteCaptionDisplay.id = 'remoteCaptionDisplay';
        remoteCaptionDisplay.className = 'caption-display';
        remoteCaptionDisplay.style.cssText = `
            position: absolute;
            bottom: 10px;
            left: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 14px;
            text-align: center;
            min-height: 25px;
            font-weight: bold;
            display: none;
            z-index: 10;
            border: 2px solid #28a745;
        `;

        videoContainer.appendChild(remoteVideo);
        videoContainer.appendChild(captionElement);
        videoContainer.appendChild(remoteCaptionDisplay);
        videoGrid.appendChild(videoContainer);
    }
}

function handleCaptionUpdate(data) {
    console.log('Caption update received:', data);
    
    // Update caption display
    const captionText = data.caption || '';
    const currentSign = data.current_sign || '';
    
    // Find the appropriate caption display element
    let captionDisplay;
    if (data.user_id === username) {
        // This is our own caption
        captionDisplay = document.getElementById('localCaptionDisplay');
    } else {
        // This is from another user
        captionDisplay = document.getElementById('remoteCaptionDisplay');
    }
    
    if (captionDisplay) {
        // Always show captions when sign language is enabled, regardless of landmarks
        if (signLanguageEnabled) {
            let displayText = '';
            if (currentSign) {
                displayText += `Detecting: ${currentSign}`;
            }
            if (captionText) {
                if (displayText) displayText += ' | ';
                displayText += `Sentence: ${captionText}`;
            }
            if (!displayText && signLanguageEnabled) {
                displayText = 'Listening for signs...';
            }
            
            captionDisplay.textContent = displayText;
            captionDisplay.style.display = 'block';
            captionDisplay.style.zIndex = '10'; // Ensure captions are above landmarks
        } else {
            captionDisplay.style.display = 'none';
        }
    }

    // Handle processed frame with landmarks (separate from captions)
    if (data.processed_frame && showLandmarks) {
        updateVideoWithLandmarks(data.user_id, data.processed_frame);
    } else if (!showLandmarks) {
        // Clear landmarks when disabled
        clearLandmarks(data.user_id);
    }
}

function updateVideoWithLandmarks(userId, processedFrameData) {
    let videoContainer;
    let videoElement;
    let canvasElement;

    if (userId === username) {
        videoContainer = localVideo.parentElement;
        videoElement = localVideo;
    } else {
        videoContainer = document.getElementById('remoteVideoContainer');
        videoElement = document.getElementById('remoteVideo');
    }

    if (!videoContainer) return;

    // Create or get canvas for landmarks
    canvasElement = videoContainer.querySelector('.landmarks-canvas');
    if (!canvasElement) {
        canvasElement = document.createElement('canvas');
        canvasElement.className = 'landmarks-canvas';
        canvasElement.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 5;
            opacity: 0.8;
        `;
        videoContainer.appendChild(canvasElement);
    }

    if (showLandmarks && processedFrameData) {
        // Create an image from the processed frame data
        const img = new Image();
        img.onload = function() {
            const ctx = canvasElement.getContext('2d');
            canvasElement.width = videoElement.videoWidth || img.width;
            canvasElement.height = videoElement.videoHeight || img.height;
            ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            
            // Draw only the landmarks, not the full frame
            ctx.globalCompositeOperation = 'source-over';
            ctx.drawImage(img, 0, 0, canvasElement.width, canvasElement.height);
        };
        img.src = processedFrameData;
        canvasElement.style.display = 'block';
    } else {
        canvasElement.style.display = 'none';
    }
}

function clearLandmarks(userId) {
    let videoContainer;
    if (userId === username) {
        videoContainer = localVideo.parentElement;
    } else {
        videoContainer = document.getElementById('remoteVideoContainer');
    }

    if (videoContainer) {
        const canvasElement = videoContainer.querySelector('.landmarks-canvas');
        if (canvasElement) {
            canvasElement.style.display = 'none';
            const ctx = canvasElement.getContext('2d');
            ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        }
    }
}

function handleSignLanguageStatusUpdate(data) {
    console.log('Sign language status update:', data);
    // Update UI based on sign language status for specific user
}

function handleLandmarksStatusUpdate(data) {
    console.log('Landmarks status update:', data);
    // Update UI based on landmarks status for specific user
}

// Toggle video
function toggleVideo() {
    if (localStream) {
        const videoTrack = localStream.getVideoTracks()[0];
        if (videoTrack) {
            videoEnabled = !videoEnabled;
            videoTrack.enabled = videoEnabled;

            if (videoEnabled) {
                toggleVideoBtn.classList.remove('inactive');
                toggleVideoBtn.classList.add('active');
            } else {
                toggleVideoBtn.classList.remove('active');
                toggleVideoBtn.classList.add('inactive');
            }
        }
    }
}

// Toggle audio
function toggleAudio() {
    if (localStream) {
        const audioTrack = localStream.getAudioTracks()[0];
        if (audioTrack) {
            audioEnabled = !audioEnabled;
            audioTrack.enabled = audioEnabled;

            if (audioEnabled) {
                toggleAudioBtn.classList.remove('inactive');
                toggleAudioBtn.classList.add('active');
            } else {
                toggleAudioBtn.classList.remove('active');
                toggleAudioBtn.classList.add('inactive');
            }
        }
    }
}

// Toggle sign language detection
function toggleSignLanguage() {
    signLanguageEnabled = !signLanguageEnabled;

    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            type: 'toggle_sign_language',
            enable: signLanguageEnabled
        }));
    }

    updateSignLanguageButton();

    if (signLanguageEnabled) {
        startFrameCapture();
        // Show caption areas immediately
        const localCaptionDisplay = document.getElementById('localCaptionDisplay');
        if (localCaptionDisplay) {
            localCaptionDisplay.textContent = 'Listening for signs...';
            localCaptionDisplay.style.display = 'block';
            localCaptionDisplay.style.zIndex = '10'; // Ensure it's above landmarks
        }
    } else {
        stopFrameCapture();
        // Clear captions when disabled
        const localCaptionDisplay = document.getElementById('localCaptionDisplay');
        const remoteCaptionDisplay = document.getElementById('remoteCaptionDisplay');
        if (localCaptionDisplay) localCaptionDisplay.style.display = 'none';
        if (remoteCaptionDisplay) remoteCaptionDisplay.style.display = 'none';
    }
}

// Update sign language button
function updateSignLanguageButton() {
    if (signLanguageEnabled) {
        toggleSignLanguageBtn.classList.remove('inactive');
        toggleSignLanguageBtn.classList.add('active');
        toggleSignLanguageBtn.title = 'Sign Language: ON';
    } else {
        toggleSignLanguageBtn.classList.remove('active');
        toggleSignLanguageBtn.classList.add('inactive');
        toggleSignLanguageBtn.title = 'Sign Language: OFF';
    }
}

// Toggle landmarks display
function toggleLandmarks() {
    showLandmarks = !showLandmarks;

    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            type: 'toggle_landmarks',
            enable: showLandmarks
        }));
    }

    updateLandmarksButton();
    
    // Hide/show existing landmarks without affecting captions
    if (!showLandmarks) {
        clearLandmarks(username);
        // Clear remote landmarks too
        const remoteVideoContainer = document.getElementById('remoteVideoContainer');
        if (remoteVideoContainer) {
            const canvasElement = remoteVideoContainer.querySelector('.landmarks-canvas');
            if (canvasElement) {
                canvasElement.style.display = 'none';
            }
        }
    }
    
    // Ensure captions remain visible if sign language is enabled
    if (signLanguageEnabled) {
        const localCaptionDisplay = document.getElementById('localCaptionDisplay');
        const remoteCaptionDisplay = document.getElementById('remoteCaptionDisplay');
        if (localCaptionDisplay && localCaptionDisplay.textContent) {
            localCaptionDisplay.style.display = 'block';
            localCaptionDisplay.style.zIndex = '10';
        }
        if (remoteCaptionDisplay && remoteCaptionDisplay.textContent) {
            remoteCaptionDisplay.style.display = 'block';
            remoteCaptionDisplay.style.zIndex = '10';
        }
    }
}

// Update landmarks button
function updateLandmarksButton() {
    if (showLandmarks) {
        toggleLandmarksBtn.classList.remove('inactive');
        toggleLandmarksBtn.classList.add('active');
        toggleLandmarksBtn.title = 'Landmarks: ON';
    } else {
        toggleLandmarksBtn.classList.remove('active');
        toggleLandmarksBtn.classList.add('inactive');
        toggleLandmarksBtn.title = 'Landmarks: OFF';
    }
}

// Start capturing frames for sign language detection
function startFrameCapture() {
    if (frameInterval) return;

    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    let frameCount = 0;

    frameInterval = setInterval(() => {
        if (!localStream || !signLanguageEnabled || !socket || socket.readyState !== WebSocket.OPEN) return;

        const videoTrack = localStream.getVideoTracks()[0];
        if (!videoTrack || !videoTrack.enabled) return;

        // Check if video has valid dimensions
        if (localVideo.videoWidth === 0 || localVideo.videoHeight === 0) return;

        // Process every 2nd frame to reduce load (inspired by import cv2 copy.py frame control)
        frameCount++;
        if (frameCount % 2 !== 0) return;

        // Set canvas dimensions to match video (reduced resolution for performance)
        const targetWidth = Math.min(localVideo.videoWidth, 640);
        const targetHeight = Math.min(localVideo.videoHeight, 480);
        
        canvas.width = targetWidth;
        canvas.height = targetHeight;

        // Draw video frame to canvas
        context.drawImage(localVideo, 0, 0, targetWidth, targetHeight);

        // Convert canvas to base64 image with reduced quality
        const frameData = canvas.toDataURL('image/jpeg', 0.7);

        // Send frame to server for processing
        socket.send(JSON.stringify({
            type: 'video_frame_for_sign',
            frame_data: frameData
        }));
    }, 1000 / frameRate);
}

// Stop capturing frames
function stopFrameCapture() {
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }
}

// End the call
function endCall() {
    if (confirm('Are you sure you want to end the call?')) {
        // Stop local stream
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
        }

        // Stop frame capture
        stopFrameCapture();

        // Close WebSocket connection
        if (socket) {
            socket.close();
        }

        // Close peer connection
        if (peerConnection) {
            peerConnection.close();
        }

        // Redirect to home
        window.location.href = '/';
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);
