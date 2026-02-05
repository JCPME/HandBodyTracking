
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
import urllib.request
import os

# Model paths
hand_model_path = os.path.expanduser("~/hand-tracking/hand_landmarker.task")
pose_model_path = os.path.expanduser("~/hand-tracking/pose_landmarker.task")

# Download models if needed
if not os.path.exists(hand_model_path):
    print("Downloading hand landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, hand_model_path)
    print("Done!")

if not os.path.exists(pose_model_path):
    print("Downloading pose landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, pose_model_path)
    print("Done!")

# Hand detector setup
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=hand_model_path),
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# Pose detector setup
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=pose_model_path),
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

# Pose connections for drawing
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]

# Hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Orbbec pipeline setup
pipeline = Pipeline()
config = Config()
profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
profile = profile_list.get_video_stream_profile(1280, 720, OBFormat.MJPG, 30)
print(f"Using: {profile.get_width()}x{profile.get_height()} @ {profile.get_fps()}fps")
config.enable_stream(profile)

pipeline.start(config)
print("Starting hand + body tracking... Press 'q' to quit")

frame_count = 0
import time
start_time = time.time()

try:
    while True:
        frames = pipeline.wait_for_frames(1000)
        if frames is None:
            continue
        
        color_frame = frames.get_color_frame()
        if color_frame is None:
            continue
        
        data = np.asanyarray(color_frame.get_data())
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        # Prepare for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Detect pose
        pose_result = pose_detector.detect(mp_image)
        
        # Detect hands
        hand_result = hand_detector.detect(mp_image)
        
        h, w = img.shape[:2]
        
        # Draw pose (green)
        if pose_result.pose_landmarks:
            for pose_landmarks in pose_result.pose_landmarks:
                for connection in POSE_CONNECTIONS:
                    start = pose_landmarks[connection[0]]
                    end = pose_landmarks[connection[1]]
                    if start.visibility > 0.5 and end.visibility > 0.5:
                        start_point = (int(start.x * w), int(start.y * h))
                        end_point = (int(end.x * w), int(end.y * h))
                        cv2.line(img, start_point, end_point, (0, 255, 0), 2)
                for landmark in pose_landmarks:
                    if landmark.visibility > 0.5:
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(img, (cx, cy), 4, (0, 200, 0), -1)
        
        # Draw hands (blue for first, red for second)
        if hand_result.hand_landmarks:
            for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                color = (255, 0, 0) if i == 0 else (0, 0, 255)
                for connection in HAND_CONNECTIONS:
                    start = hand_landmarks[connection[0]]
                    end = hand_landmarks[connection[1]]
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(img, start_point, end_point, color, 2)
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(img, (cx, cy), 5, color, -1)
        
        # FPS counter
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow('Hand + Body Tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
