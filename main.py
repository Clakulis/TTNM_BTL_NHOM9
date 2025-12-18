import customtkinter as ctk
from threading import Thread
from datetime import datetime
import os
import cv2
from PIL import Image, ImageTk
import csv
import time
import socketio
from minio import Minio  # pip install minio
from detectors import EmotionDetector, MediaPipeDrowsinessDetector
from pygame import mixer
import faulthandler
# Import the new PDF generator
from view_results import view_results_from_csv, generate_pdf_report 

faulthandler.enable()

# === CONFIGURATION ===
MODEL_PATH = 'model.h5'
HAAR_PATH = "haarcascade_frontalface_default.xml"
SERVER_URL = "http://127.0.0.1:5000"

# === MINIO CONFIGURATION ===
MINIO_ENDPOINT = "127.0.0.1:9000"  # Localhost Docker
MINIO_ACCESS_KEY = "admin"         # Matches docker-compose environment
MINIO_SECRET_KEY = "password123"   # Matches docker-compose environment
MINIO_BUCKET = "student-reports"
SECURE_CONNECTION = False          # IMPORTANT: Local Docker uses HTTP, not HTTPS

# SOUNDS
SOUND_WAKE = 'wake_up.mp3'
SOUND_ALERT = 'alert.mp3'
SOUND_MISSING = 'Where Are You, Sir_.mp3'

class FocusTrackerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Focus & Emotion Tracker Pro (Minio Integrated)")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}+0+0")
        
        self.init_logic()
        self.init_ui()

    def init_logic(self):
        self.app_running = False
        self.paused = False
        self.student_id = "unknown"
        
        # WEBSOCKET CLIENT
        self.sio = socketio.Client()
        
        # MINIO CLIENT
        try:
            self.minio_client = Minio(
                MINIO_ENDPOINT,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=SECURE_CONNECTION
            )
            # Check/Create Bucket
            if not self.minio_client.bucket_exists(MINIO_BUCKET):
                self.minio_client.make_bucket(MINIO_BUCKET)
        except Exception as e:
            print(f"Minio Connection Failed: {e}")
            self.minio_client = None

        # State variables
        self.current_emotion = "Neutral"
        self.is_drowsy = False
        self.is_yawning = False
        self.face_detected = False
        self.ear = 0.0
        self.mar = 0.0
        
        # Load Models & Sounds
        try:
            self.emotion_detector = EmotionDetector(MODEL_PATH)
            self.drowsiness_detector = MediaPipeDrowsinessDetector()
            self.face_cascade = cv2.CascadeClassifier(HAAR_PATH)
            mixer.init()
            self.sound_wake = mixer.Sound(SOUND_WAKE)
            self.sound_alert = mixer.Sound(SOUND_ALERT)
            self.sound_missing = mixer.Sound(SOUND_MISSING)
        except Exception as e:
            print(f"Init Error: {e}")
            self.sound_wake = self.sound_alert = self.sound_missing = None

        self.alarm_active = False
        self.missing_alarm_active = False
        self.face_missing_counter = 0
        self.FACE_MISSING_THRESHOLD = 50

    def init_ui(self):
        self.grid_columnconfigure(0, weight=3) 
        self.grid_columnconfigure(1, weight=1) 
        self.grid_rowconfigure(0, weight=1)

        self.video_frame = ctk.CTkFrame(self, corner_radius=10)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_frame, text="", corner_radius=10)
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        self.stats_frame = ctk.CTkFrame(self, corner_radius=10)
        self.stats_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.lbl_title = ctk.CTkLabel(self.stats_frame, text="SESSION DASHBOARD", font=("Arial", 20, "bold"))
        self.lbl_title.pack(pady=20)

        self.entry_id = ctk.CTkEntry(self.stats_frame, placeholder_text="Enter Student ID")
        self.entry_id.pack(fill="x", padx=20, pady=5)

        self.btn_start = ctk.CTkButton(self.stats_frame, text="Start Session", command=self.start_session, fg_color="#4CAF50")
        self.btn_start.pack(pady=10, fill="x", padx=20)

        self.btn_stop = ctk.CTkButton(self.stats_frame, text="End & Upload", command=self.stop_session, state="disabled", fg_color="#F44336")
        self.btn_stop.pack(pady=10, fill="x", padx=20)

        self.btn_view = ctk.CTkButton(self.stats_frame, text="View Last Report", command=self.open_results, fg_color="transparent", border_width=1, text_color=("gray10", "#DCE4EE"))
        self.btn_view.pack(side="bottom", pady=20)

        self.lbl_status = ctk.CTkLabel(self.stats_frame, text="Status: IDLE", font=("Arial", 16, "bold"), text_color="gray")
        self.lbl_status.pack(pady=20)
        
        self.lbl_emotion = ctk.CTkLabel(self.stats_frame, text="Emotion: -", font=("Arial", 14))
        self.lbl_emotion.pack(pady=5)

    def connect_server(self):
        """Attempts to connect to the WebSocket server."""
        try:
            if not self.sio.connected:
                print(f"🔄 Attempting to connect to {SERVER_URL}...")
                self.sio.connect(SERVER_URL, wait=False) # wait=False prevents freezing UI
                print(f"✅ Connected!")
        except Exception as e:
            print(f"❌ Connection Failed: {e}")

    def run_heartbeat(self):
        """Main loop that ensures connection and sends data."""
        while self.app_running:
            if not self.paused:
                # 1. Auto-Reconnect if disconnected
                if not self.sio.connected:
                    self.connect_server()
                
                # 2. Send Data if connected
                if self.sio.connected:
                    try:
                        payload = {
                            "student_id": self.student_id,
                            "emotion": self.current_emotion,
                            "is_drowsy": bool(self.is_drowsy),
                            "is_yawning": bool(self.is_yawning),
                            "face_detected": bool(self.face_detected)
                        }
                        self.sio.emit('student_heartbeat', payload)
                    except Exception as e:
                        print(f"⚠️ Emit Error: {e}")
            
            # 3. Wait before next heartbeat
            time.sleep(0.5)

    def start_session(self):
        input_id = self.entry_id.get().strip()
        if not input_id:
            self.lbl_status.configure(text="⚠️ ENTER ID FIRST", text_color="orange")
            return

        self.student_id = input_id
        self.app_running = True
        
        self.entry_id.configure(state="disabled")
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_view.configure(state="disabled")
        self.lbl_status.configure(text="STARTING...", text_color="#4CAF50")

        # Create Log File
        os.makedirs("logs", exist_ok=True)
        # We save this timestamp to use in the filename later
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join("logs", f"{self.student_id}_{self.session_timestamp}.csv")
        self.log_file = open(self.log_path, mode="w", newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(["Timestamp", "Emotion", "EAR", "YawnDistance", "DrowsinessAlert", "YawnAlert", "FaceMissing"])
        
        self.vs = cv2.VideoCapture(0)
        Thread(target=self.run_heartbeat, daemon=True).start()
        self.update_loop()

    def stop_session(self):
        self.app_running = False
        
        if hasattr(self, 'vs') and self.vs:
            self.vs.release()

        # Close CSV so we can read it for PDF generation
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

        self.lbl_status.configure(text="GENERATING PDF...", text_color="orange")
        self.update_idletasks() # Force UI update

        # --- UPLOAD LOGIC ---
        if self.minio_client:
            # 1. Define PDF Path
            pdf_filename = f"{self.student_id}_{self.session_timestamp}.pdf"
            pdf_path = os.path.join("logs", pdf_filename)
            
            # 2. Generate PDF
            success = generate_pdf_report(self.log_path, pdf_path)
            
            if success:
                self.lbl_status.configure(text="UPLOADING...", text_color="blue")
                self.update_idletasks()
                try:
                    # 3. Upload to Minio
                    self.minio_client.fput_object(
                        MINIO_BUCKET, 
                        pdf_filename, 
                        pdf_path,
                        content_type="application/pdf"
                    )
                    print(f"Uploaded: {pdf_filename}")
                    self.lbl_status.configure(text="✅ UPLOAD COMPLETE", text_color="green")
                except Exception as e:
                    print(f"Upload Failed: {e}")
                    self.lbl_status.configure(text="❌ UPLOAD FAILED", text_color="red")
            else:
                self.lbl_status.configure(text="❌ PDF ERROR", text_color="red")
        else:
             self.lbl_status.configure(text="⚠️ NO MINIO", text_color="orange")
        
        self.video_label.configure(image="")
        self.entry_id.configure(state="normal")
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_view.configure(state="normal")

        if self.sio.connected:
            self.sio.disconnect()

    def open_results(self):
        if hasattr(self, 'log_path') and os.path.exists(self.log_path):
            result_window = ctk.CTkToplevel(self)
            result_window.title("Session Results")
            result_window.geometry("1000x800")
            res_frame = ctk.CTkFrame(result_window)
            res_frame.pack(fill="both", expand=True)
            view_results_from_csv(res_frame, self.log_path)

    def update_loop(self):
        if not self.app_running: return

        ret, frame = self.vs.read()
        if not ret:
            self.after(10, self.update_loop)
            return

        target_w = self.video_label.winfo_width()
        target_h = self.video_label.winfo_height()
        if target_w <= 1: target_w, target_h = 640, 480
        
        h, w = frame.shape[:2]
        aspect = w/h
        if (target_w/target_h) > aspect:
            new_h = target_h
            new_w = int(aspect*target_h)
        else:
            new_w = target_w
            new_h = int(target_w/aspect)
        frame = cv2.resize(frame, (new_w, new_h))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        self.current_emotion = "Neutral"
        self.is_drowsy = False
        self.is_yawning = False
        self.ear = 0.0
        self.mar = 0.0

        if len(faces) == 0:
            self.face_detected = False
            self.face_missing_counter += 1
            if self.face_missing_counter >= self.FACE_MISSING_THRESHOLD:
                if not self.missing_alarm_active and self.sound_missing:
                    self.missing_alarm_active = True
                    Thread(target=self.sound_missing.play, daemon=True).start()
                cv2.putText(frame, "FACE MISSING", (50, new_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            self.lbl_status.configure(text="❓ NO FACE", text_color="gray")
        else:
            self.face_detected = True
            self.face_missing_counter = 0
            self.missing_alarm_active = False
            
            (x, y, w_f, h_f) = faces[0]
            face_roi = gray[y:y+h_f, x:x+w_f]
            
            self.current_emotion, _, _ = self.emotion_detector.detect_emotion(face_roi)
            self.is_drowsy, self.is_yawning, self.ear, self.mar, landmarks = self.drowsiness_detector.detect_drowsiness(frame)

            color = (0, 255, 0)
            if self.is_drowsy: color = (0, 0, 255)
            elif self.is_yawning: color = (0, 165, 255)
            
            cv2.rectangle(frame, (x, y), (x+w_f, y+h_f), color, 2)
            cv2.putText(frame, self.current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            self.lbl_emotion.configure(text=f"Emotion: {self.current_emotion}")
            
            if self.is_drowsy:
                self.lbl_status.configure(text="⚠️ DROWSY!", text_color="red")
                cv2.putText(frame, "DROWSY!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                if not self.alarm_active and self.sound_wake:
                    self.alarm_active = True
                    Thread(target=self.sound_wake.play, daemon=True).start()
            elif self.is_yawning:
                self.lbl_status.configure(text="⚠️ YAWNING!", text_color="orange")
                cv2.putText(frame, "YAWNING!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 3)
                if not self.alarm_active and self.sound_alert:
                    self.alarm_active = True
                    Thread(target=self.sound_alert.play, daemon=True).start()
            else:
                self.lbl_status.configure(text="✅ TRACKING", text_color="#4CAF50")
                self.alarm_active = False

        # LOG TO CSV
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_writer.writerow([ts, self.current_emotion, f"{self.ear:.2f}", f"{self.mar:.2f}", 
                                  "Yes" if self.is_drowsy else "No", "Yes" if self.is_yawning else "No", 
                                  "No" if self.face_detected else "Yes"])
        self.log_file.flush()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.after(15, self.update_loop)

if __name__ == "__main__":
    app = FocusTrackerApp()
    app.mainloop()