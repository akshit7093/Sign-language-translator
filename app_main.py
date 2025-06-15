import asyncio
import base64
import cv2
import numpy as np
import uvicorn
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, List, Set, DefaultDict
from collections import defaultdict
import os
import traceback
import time

# Import the sign language processing module
try:
    from sign_language_module import SignDetector, ACTIONS, draw_styled_landmarks, prob_viz, PROB_COLORS
    print("Sign language module imported successfully")
except Exception as e:
    print(f"Error importing sign language module: {e}")
    traceback.print_exc()
    # Create dummy implementations to prevent crashes
    class SignDetector:
        def __init__(self):
            pass
        def process_frame(self, frame):
            return "", "", None
    
    ACTIONS = ["Hello", "Thank you", "I love you"]
    def draw_styled_landmarks(image, results):
        pass
    def prob_viz(res, actions, input_frame, colors):
        return input_frame
    PROB_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# --- FastAPI App Initialization ---
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates for serving HTML
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- Data Structures for Room and User Management ---
class Room:
    def __init__(self, room_id: str):
        self.room_id: str = room_id
        self.participants: Dict[WebSocket, str] = {}  # WebSocket: user_id
        self.user_sign_detectors: Dict[str, SignDetector] = {} # user_id: SignDetector instance
        self.sign_language_enabled_users: Set[str] = set() # user_id
        self.show_landmarks_enabled_users: Set[str] = set()
        self.user_info: Dict[str, Dict[str, str]] = {}
        self.last_frame_time: Dict[str, float] = {}  # Track frame processing time per user

    async def add_participant(self, websocket: WebSocket, user_id: str, username: str):
        self.participants[websocket] = user_id
        self.user_sign_detectors[user_id] = SignDetector() # Each user gets their own detector instance
        self.user_info[user_id] = {'username': username}
        self.last_frame_time[user_id] = 0
        print(f"User {username} ({user_id}) joined room {self.room_id}")
        await self.broadcast_user_list()

    async def remove_participant(self, websocket: WebSocket):
        user_id = self.participants.pop(websocket, None)
        if user_id:
            self.user_sign_detectors.pop(user_id, None)
            self.sign_language_enabled_users.discard(user_id)
            self.show_landmarks_enabled_users.discard(user_id)
            self.user_info.pop(user_id, None)
            self.last_frame_time.pop(user_id, None)
            print(f"User ({user_id}) left room {self.room_id}")
            await self.broadcast_user_list()
        return user_id

    async def broadcast(self, message: dict, exclude_sender: WebSocket = None):
        disconnected_sockets = []
        for participant_ws in self.participants:
            if participant_ws != exclude_sender:
                try:
                    await participant_ws.send_json(message)
                except WebSocketDisconnect:
                    disconnected_sockets.append(participant_ws)
                except Exception as e:
                    print(f"Error broadcasting to a participant: {e}")
                    disconnected_sockets.append(participant_ws)
        
        # Clean up disconnected sockets
        for ws in disconnected_sockets:
            await self.remove_participant(ws)
    
    async def broadcast_user_list(self):
        user_list = []
        for ws, uid in self.participants.items():
            user_info = self.user_info.get(uid, {})
            user_list.append({
                'user_id': uid,
                'username': user_info.get('username', uid)
            })
        await self.broadcast({'type': 'user_list', 'users': user_list})

    def get_username_by_id(self, user_id_to_find: str) -> str:
        return self.user_info.get(user_id_to_find, {}).get('username', user_id_to_find)

rooms: Dict[str, Room] = {}

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.user_ws_map: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, room_id: str, user_id: str):
        print(f"Attempting to accept WebSocket for user {user_id} in room {room_id}")
        await websocket.accept()
        print(f"WebSocket accepted for user {user_id} in room {room_id}")
        
        if room_id not in rooms:
            rooms[room_id] = Room(room_id)
        
        username = user_id
        await rooms[room_id].add_participant(websocket, user_id, username)
        self.active_connections[room_id].add(websocket)
        self.user_ws_map[user_id] = websocket
        print(f"User {username} ({user_id}) connected to room {room_id}. Total in room: {len(rooms[room_id].participants)}")

    async def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in rooms:
            self.active_connections[room_id].discard(websocket)
            user_id = await rooms[room_id].remove_participant(websocket)
            if user_id and user_id in self.user_ws_map:
                del self.user_ws_map[user_id]
            if not rooms[room_id].participants:
                print(f"Room {room_id} is empty, removing.")
                del rooms[room_id]
            print(f"User ({user_id}) disconnected from room {room_id}.")

    async def broadcast_to_room(self, room_id: str, message: dict, exclude_sender: WebSocket = None):
        if room_id in rooms:
            await rooms[room_id].broadcast(message, exclude_sender)

manager = ConnectionManager()

# --- FastAPI HTTP Routes ---
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("simple_video_call.html", {"request": request})

# --- FastAPI WebSocket Endpoints ---
@app.websocket("/ws/{room_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, user_id: str):
    await manager.connect(websocket, room_id, user_id)
    client_ip = websocket.client.host if websocket.client else "Unknown IP"
    print(f"WebSocket connection established for user {user_id} in room {room_id} from {client_ip}")

    try:
        print(f"Entering WebSocket receive loop for user {user_id}")
        while True:
            data = await websocket.receive_json()
            current_room = rooms.get(room_id)
            if not current_room:
                print(f"Room {room_id} not found for message from {user_id}")
                break

            message_type = data.get("type")

            if message_type == "webrtc_signal":
                recipient_id = data.get("to")
                if recipient_id and recipient_id in manager.user_ws_map:
                    await manager.user_ws_map[recipient_id].send_json(data)
                else:
                     await manager.broadcast_to_room(room_id, data, exclude_sender=websocket)

            elif message_type == "toggle_sign_language":
                enable = data.get("enable", False)
                if enable:
                    current_room.sign_language_enabled_users.add(user_id)
                else:
                    current_room.sign_language_enabled_users.discard(user_id)
                print(f"User {user_id} in room {room_id} set sign language to {enable}")
                await manager.broadcast_to_room(room_id, {
                    "type": "sign_language_status_update",
                    "user_id": user_id,
                    "enabled": enable
                })

            elif message_type == "toggle_landmarks":
                enable = data.get("enable", False)
                if enable:
                    current_room.show_landmarks_enabled_users.add(user_id)
                else:
                    current_room.show_landmarks_enabled_users.discard(user_id)
                print(f"User {user_id} in room {room_id} set landmarks to {enable}")
                await manager.broadcast_to_room(room_id, {
                    "type": "landmarks_status_update",
                    "user_id": user_id,
                    "enabled": enable
                })

            elif message_type == "video_frame_for_sign":
                if user_id in current_room.sign_language_enabled_users:
                    base64_frame = data.get("frame_data")
                    if base64_frame:
                        try:
                            # Decode base64 frame
                            img_bytes = base64.b64decode(base64_frame.split(',')[-1])
                            np_arr = np.frombuffer(img_bytes, np.uint8)
                            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                            if frame is not None:
                                detector = current_room.user_sign_detectors.get(user_id)
                                if detector:
                                    current_sign, sentence, raw_results = detector.process_frame(frame.copy())
                                    
                                    processed_frame_base64 = None
                                    if user_id in current_room.show_landmarks_enabled_users and raw_results:
                                        # Draw landmarks on a copy of the frame
                                        landmarks_frame = frame.copy()
                                        draw_styled_landmarks(landmarks_frame, raw_results)
                                        _, buffer = cv2.imencode('.jpg', landmarks_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                        processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
                                        processed_frame_base64 = f"data:image/jpeg;base64,{processed_frame_base64}"

                                    # Always send caption update when sign language is enabled
                                    caption_message = {
                                        "type": "caption_update",
                                        "user_id": user_id,
                                        "username": current_room.get_username_by_id(user_id),
                                        "caption": sentence,
                                        "current_sign": current_sign,
                                        "processed_frame": processed_frame_base64  # Only included if landmarks enabled
                                    }
                                    await manager.broadcast_to_room(room_id, caption_message)
                                    
                                    # Debug print
                                    if current_sign or sentence:
                                        print(f"Sent caption update for {user_id}: current='{current_sign}', sentence='{sentence}', landmarks={'enabled' if processed_frame_base64 else 'disabled'}")
                            else:
                                print(f"Failed to decode frame for user {user_id}")
                        except Exception as e:
                            print(f"Error processing frame for sign detection for user {user_id}: {e}")
                            traceback.print_exc()

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for user {user_id} in room {room_id} from {client_ip}")
    except Exception as e:
        print(f"Error in WebSocket for user {user_id} in room {room_id}: {e}")
        traceback.print_exc()
    finally:
        await manager.disconnect(websocket, room_id)
        print(f"Cleaned up connection for user {user_id} in room {room_id}")

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting Uvicorn server. Ensure 'action.h5' is accessible by 'sign_language_module.py'.")
    print(f"Expected model path by sign_language_module: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'action.h5')}")
    print(f"Actions configured in sign_language_module: {ACTIONS}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
