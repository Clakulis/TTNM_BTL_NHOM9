import customtkinter as ctk
from tkinter import ttk
from threading import Thread
from datetime import datetime
import os
import cv2
from PIL import Image, ImageTk
import csv
import numpy as np
import time
from view_results import view_results_from_csv, export_figure_to_pdf
from detectors import EmotionDetector, DrowsinessDetector, MediaPipeDrowsinessDetector
from pygame import mixer
import dlib

# === CONFIGURATION ===
# Paths
MODEL_PATH = 'model.h5'
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
HAAR_PATH = "haarcascade_frontalface_default.xml"

# Sounds
SOUND_WAKE = 'wake_up.mp3'
SOUND_ALERT = 'alert.mp3'
SOUND_MISSING = 'Where Are You, Sir_.mp3'

# === APP CLASS ===
class FocusTrackerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Setup Window
        self.title("Focus & Emotion Tracker Pro")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}+0+0")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Initialize Logic
        self.init_logic()
        
        # Initialize UI
        self.init_ui()

    def init_logic(self):
        # State
        self.app_running = False
        self.paused = False
        self.start_time = None
        self.pause_start = None
        self.paused_time = 0
        self.log_file = None
        self.log_writer = None
        self.vs = None
        
        # Detectors
        try:
            self.emotion_detector = EmotionDetector(MODEL_PATH)
            # self.drowsiness_detector = DrowsinessDetector(PREDICTOR_PATH) # Old Dlib detector
            self.drowsiness_detector = MediaPipeDrowsinessDetector() # New MediaPipe detector
            self.face_cascade = cv2.CascadeClassifier(HAAR_PATH)
        except Exception as e:
            print(f"Error loading models: {e}")
            # Handle error gracefully (maybe show message box)

        # Audio
        mixer.init()
        try:
            self.sound_wake = mixer.Sound(SOUND_WAKE)
            self.sound_alert = mixer.Sound(SOUND_ALERT)
            self.sound_missing = mixer.Sound(SOUND_MISSING)
        except Exception as e:
            print(f"Warning: Sound files not found or error loading: {e}")
            self.sound_wake = None
            self.sound_alert = None
            self.sound_missing = None

        self.alarm_active = False
        self.missing_alarm_active = False

        # Counters
        self.face_missing_counter = 0
        self.FACE_MISSING_THRESHOLD = 50

    def init_ui(self):
        # Main Layout: 2 Columns (Video | Stats)
        self.grid_columnconfigure(0, weight=3) # Video area
        self.grid_columnconfigure(1, weight=1) # Stats area
        self.grid_rowconfigure(0, weight=1)

        # === LEFT PANEL: VIDEO ===
        self.video_frame = ctk.CTkFrame(self, corner_radius=10)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="", corner_radius=10)
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # === RIGHT PANEL: CONTROLS & STATS ===
        self.stats_frame = ctk.CTkFrame(self, corner_radius=10)
        self.stats_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Title
        self.lbl_title = ctk.CTkLabel(self.stats_frame, text="SESSION DASHBOARD", font=("Arial", 20, "bold"))
        self.lbl_title.pack(pady=20)

        # Timer
        self.lbl_timer = ctk.CTkLabel(self.stats_frame, text="00:00", font=("Arial", 40, "bold"), text_color="#4CAF50")
        self.lbl_timer.pack(pady=10)

        # Controls
        self.btn_start = ctk.CTkButton(self.stats_frame, text="Start Session", command=self.start_session, height=40, font=("Arial", 14))
        self.btn_start.pack(pady=10, fill="x", padx=20)

        self.btn_pause = ctk.CTkButton(self.stats_frame, text="Pause", command=self.pause_session, state="disabled", fg_color="#FFC107", text_color="black", height=40, font=("Arial", 14))
        self.btn_pause.pack(pady=10, fill="x", padx=20)

        self.btn_stop = ctk.CTkButton(self.stats_frame, text="End Session", command=self.end_session, state="disabled", fg_color="#F44336", height=40, font=("Arial", 14))
        self.btn_stop.pack(pady=10, fill="x", padx=20)

        # Stats Display
        self.stats_container = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        self.stats_container.pack(fill="both", expand=True, padx=10, pady=10)

        self.lbl_emotion = ctk.CTkLabel(self.stats_container, text="Emotion: -", font=("Arial", 16))
        self.lbl_emotion.pack(anchor="w", pady=5)
        
        self.lbl_ear = ctk.CTkLabel(self.stats_container, text="EAR (Eyes): -", font=("Arial", 16))
        self.lbl_ear.pack(anchor="w", pady=5)

        self.lbl_mar = ctk.CTkLabel(self.stats_container, text="MAR (Mouth): -", font=("Arial", 16))
        self.lbl_mar.pack(anchor="w", pady=5)

        self.lbl_status = ctk.CTkLabel(self.stats_container, text="Status: READY", font=("Arial", 18, "bold"), text_color="gray")
        self.lbl_status.pack(pady=20)

        # View Results Button (Bottom)
        self.btn_view_results = ctk.CTkButton(self.stats_frame, text="View Last Report", command=self.open_results, fg_color="transparent", border_width=1)
        self.btn_view_results.pack(side="bottom", pady=20)

    def start_session(self):
        if self.app_running: return
        
        self.app_running = True
        self.paused = False
        self.start_time = time.time()
        
        # UI Updates
        self.btn_start.configure(state="disabled")
        self.btn_pause.configure(state="normal")
        self.btn_stop.configure(state="normal")
        self.lbl_status.configure(text="TRACKING ACTIVE", text_color="#4CAF50")

        # Logging
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join("logs", f"emotion_log_{timestamp}.csv")
        self.log_file = open(self.log_path, mode="w", newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(["Timestamp", "Emotion", "EAR", "YawnDistance", "DrowsinessAlert", "YawnAlert", "FaceMissing"])

        # Camera
        self.vs = cv2.VideoCapture(0)
        
        # Start Loop
        self.update_loop()
        self.update_timer()

    def pause_session(self):
        if not self.app_running: return
        self.paused = not self.paused
        if self.paused:
            self.pause_start = time.time()
            self.btn_pause.configure(text="Resume", fg_color="#4CAF50")
            self.lbl_status.configure(text="PAUSED", text_color="#FFC107")
        else:
            self.start_time += (time.time() - self.pause_start)
            self.btn_pause.configure(text="Pause", fg_color="#FFC107")
            self.lbl_status.configure(text="TRACKING ACTIVE", text_color="#4CAF50")

    def end_session(self):
        self.app_running = False
        if self.vs: self.vs.release()
        if self.log_file: self.log_file.close()
        
        self.btn_start.configure(state="normal")
        self.btn_pause.configure(state="disabled", text="Pause")
        self.btn_stop.configure(state="disabled")
        self.lbl_status.configure(text="SESSION ENDED", text_color="gray")
        self.video_label.configure(image="")
        
        # Open results automatically or just enable button
        self.open_results()

    def open_results(self):
        if hasattr(self, 'log_path') and os.path.exists(self.log_path):
            # Create a new Toplevel window for results
            result_window = ctk.CTkToplevel(self)
            result_window.title("Session Results")
            result_window.geometry("1000x800")
            
            res_frame = ctk.CTkFrame(result_window)
            res_frame.pack(fill="both", expand=True)
            view_results_from_csv(res_frame, self.log_path)

    def update_timer(self):
        if not self.app_running: return
        if not self.paused:
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            self.lbl_timer.configure(text=f"{mins:02}:{secs:02}")
        self.after(1000, self.update_timer)

    def update_loop(self):
        if not self.app_running: return
        if self.paused:
            self.after(100, self.update_loop)
            return

        ret, frame = self.vs.read()
        if not ret:
            self.after(10, self.update_loop)
            return

        # Process Frame
        frame = cv2.resize(frame, (640, 480)) # Better resolution
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        current_emotion = "None"
        ear = 0.0
        mar = 0.0
        is_drowsy = False
        is_yawning = False
        
        if len(faces) == 0:
            self.face_missing_counter += 1
            if self.face_missing_counter >= self.FACE_MISSING_THRESHOLD:
                if not self.missing_alarm_active and self.sound_missing:
                    self.missing_alarm_active = True
                    Thread(target=self.sound_missing.play, daemon=True).start()
                
                # Visual Alert for Missing Face
                cv2.putText(frame, "FACE NOT DETECTED!", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            self.face_missing_counter = 0
            self.missing_alarm_active = False
            
            # Process first face only for focus tracking
            (x, y, w, h) = faces[0]
            
            # 1. Emotion Detection
            face_roi = gray[y:y+h, x:x+w]
            current_emotion, raw_emotion, preds = self.emotion_detector.detect_emotion(face_roi)
            
            # 2. Drowsiness Detection (MediaPipe)
            # MediaPipe processes the whole frame and finds faces itself, but we can pass the frame.
            # Note: MediaPipe might find a different face than Haar Cascade if multiple faces are present.
            # Ideally, we should use MediaPipe for face detection too, but to keep structure similar:
            is_drowsy, is_yawning, ear, mar, landmarks_np = self.drowsiness_detector.detect_drowsiness(frame)
            
            # 3. Draw UI on Frame
            # Face Box (Haar Cascade)
            color = (0, 255, 0) if not is_drowsy else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Landmarks (MediaPipe)
            if landmarks_np is not None:
                # Draw only eyes and mouth to keep it clean
                for idx in self.drowsiness_detector.LEFT_EYE + self.drowsiness_detector.RIGHT_EYE:
                    cv2.circle(frame, tuple(landmarks_np[idx]), 1, (0, 255, 255), -1)
                
                # Draw Mouth
                mouth_indices = [self.drowsiness_detector.MOUTH_TOP, self.drowsiness_detector.MOUTH_BOTTOM, 
                                 self.drowsiness_detector.MOUTH_LEFT, self.drowsiness_detector.MOUTH_RIGHT]
                for idx in mouth_indices:
                    cv2.circle(frame, tuple(landmarks_np[idx]), 2, (0, 0, 255), -1)

            # Emotion Label
            cv2.putText(frame, current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            # Alerts
            if is_drowsy:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                if not self.alarm_active and self.sound_wake:
                    self.alarm_active = True
                    Thread(target=self.sound_wake.play, daemon=True).start()
            elif is_yawning:
                cv2.putText(frame, "YAWN DETECTED!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                if not self.alarm_active and self.sound_alert:
                    self.alarm_active = True
                    Thread(target=self.sound_alert.play, daemon=True).start()
            else:
                self.alarm_active = False

        # Update Stats Panel
        self.lbl_emotion.configure(text=f"Emotion: {current_emotion}")
        self.lbl_ear.configure(text=f"EAR: {ear:.2f} {'(LOW)' if ear < 0.25 and ear > 0 else ''}", text_color="red" if ear < 0.25 and ear > 0 else "white")
        self.lbl_mar.configure(text=f"MAR: {mar:.2f}")
        
        if is_drowsy:
            self.lbl_status.configure(text="⚠️ DROWSY!", text_color="red")
        elif is_yawning:
            self.lbl_status.configure(text="⚠️ YAWNING!", text_color="orange")
        elif len(faces) == 0:
             self.lbl_status.configure(text="❓ NO FACE", text_color="gray")
        else:
            self.lbl_status.configure(text="✅ FOCUSED", text_color="#4CAF50")

        # Log Data
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_writer.writerow([ts, current_emotion, f"{ear:.2f}", f"{mar:.2f}", "Yes" if is_drowsy else "No", "Yes" if is_yawning else "No", "Yes" if len(faces)==0 else "No"])

        # Convert to Tkinter Image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.after(10, self.update_loop)

if __name__ == "__main__":
    app = FocusTrackerApp()
    app.mainloop()
