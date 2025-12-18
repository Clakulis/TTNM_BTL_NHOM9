import cv2
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from scipy.spatial import distance as dist
from collections import deque, Counter
import mediapipe as mp

class EmotionDetector:
    def __init__(self, model_path, labels=None, smooth_window=3):
        self.model = load_model(model_path)
        self.labels = labels if labels else ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.emotion_window = deque(maxlen=smooth_window)

    def detect_emotion(self, face_roi):
        """
        Predicts emotion from a face ROI (grayscale).
        Returns: (smoothed_label, raw_label, probability_dist)
        """
        try:
            roi = cv2.resize(face_roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = self.model.predict(roi, verbose=0)[0]
            raw_label = self.labels[preds.argmax()]
            
            # Add to window for smoothing
            self.emotion_window.append(raw_label)
            
            # Voting mechanism
            smoothed_label = Counter(self.emotion_window).most_common(1)[0][0]
            
            return smoothed_label, raw_label, preds
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return "Neutral", "Neutral", np.zeros(len(self.labels))

class MediaPipeDrowsinessDetector:
    def __init__(self, ear_thresh=0.20, ear_frames=10, mar_thresh=0.5, smile_thresh=0.3, yawn_frames=15):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Thresholds
        self.EAR_THRESH = ear_thresh
        self.EAR_CONSEC_FRAMES = ear_frames
        self.MAR_THRESH = mar_thresh # Mouth Aspect Ratio threshold for Yawning
        self.SMILE_THRESH = smile_thresh # Mouth Aspect Ratio threshold for Smiling (Low MAR)
        self.YAWN_CONSEC_FRAMES = yawn_frames
        
        # Counters
        self.COUNTER = 0
        self.YAWN_COUNTER = 0

        # Indices for MediaPipe (approximate)
        # Left Eye
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        # Right Eye
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        # Mouth (Top, Bottom, Left, Right)
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291

    def _euclidean_distance(self, point1, point2):
        x1, y1 = point1.ravel()
        x2, y2 = point2.ravel()
        return dist.euclidean((x1, y1), (x2, y2))

    def _eye_aspect_ratio(self, landmarks, indices):
        # landmarks is a list of (x, y) tuples
        # indices: [p1, p2, p3, p4, p5, p6]
        # Vertical 1: p2 - p6
        # Vertical 2: p3 - p5
        # Horizontal: p1 - p4
        
        p2 = landmarks[indices[1]]
        p6 = landmarks[indices[5]]
        p3 = landmarks[indices[2]]
        p5 = landmarks[indices[4]]
        p1 = landmarks[indices[0]]
        p4 = landmarks[indices[3]]

        A = self._euclidean_distance(p2, p6)
        B = self._euclidean_distance(p3, p5)
        C = self._euclidean_distance(p1, p4)

        return (A + B) / (2.0 * C)

    def _mouth_aspect_ratio(self, landmarks):
        # Vertical: Top - Bottom
        # Horizontal: Left - Right
        top = landmarks[self.MOUTH_TOP]
        bottom = landmarks[self.MOUTH_BOTTOM]
        left = landmarks[self.MOUTH_LEFT]
        right = landmarks[self.MOUTH_RIGHT]

        vertical = self._euclidean_distance(top, bottom)
        horizontal = self._euclidean_distance(left, right)

        # Avoid division by zero
        if horizontal == 0:
            return 0
        return vertical / horizontal

    def detect_drowsiness(self, frame):
        """
        Checks for drowsiness and yawning using MediaPipe.
        Returns: (is_drowsy, is_yawning, ear_value, mar_value, landmarks_np)
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        is_drowsy = False
        is_yawning = False
        ear = 0
        mar = 0
        landmarks_np = None

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to numpy array (x, y)
            h, w, _ = frame.shape
            landmarks_np = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])

            # 1. Calculate EAR
            leftEAR = self._eye_aspect_ratio(landmarks_np, self.LEFT_EYE)
            rightEAR = self._eye_aspect_ratio(landmarks_np, self.RIGHT_EYE)
            ear = (leftEAR + rightEAR) / 2.0

            # 2. Calculate MAR
            mar = self._mouth_aspect_ratio(landmarks_np)

            # Check Drowsiness
            # Condition: EAR is low AND User is NOT smiling (MAR is not too low)
            # Smiling widens the mouth horizontally, lowering MAR.
            is_smiling = mar < self.SMILE_THRESH
            
            if ear < self.EAR_THRESH and not is_smiling:
                self.COUNTER += 1
                if self.COUNTER >= self.EAR_CONSEC_FRAMES:
                    is_drowsy = True
            else:
                self.COUNTER = 0
                is_drowsy = False

            # Check Yawning
            # Using MAR (Height/Width) helps distinguish Yawn (High Ratio) from Smile (Low Ratio)
            if mar > self.MAR_THRESH:
                self.YAWN_COUNTER += 1
                if self.YAWN_COUNTER >= self.YAWN_CONSEC_FRAMES:
                    is_yawning = True
            else:
                self.YAWN_COUNTER = 0
                is_yawning = False
        
        return is_drowsy, is_yawning, ear, mar, landmarks_np

