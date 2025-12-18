from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from datetime import datetime
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- DATABASE ---
students_db = {}
MAX_HISTORY_POINTS = 5000

# --- GRADING LOGIC ---
EMOTION_SCORE_MAP = {
    'Happy': 2, 'Surprise': 2, 'Neutral': 1,
    'Sad': 0, 'Angry': -1, 'Fear': -2, 'Disgust': -3
}

def calculate_focus_score(emotion, face_detected, is_drowsy, is_yawning):
    if not face_detected: return -1
    if is_drowsy: return -3
    if is_yawning: return -2
    return EMOTION_SCORE_MAP.get(emotion, 0)

@app.route('/')
def teacher_ui():
    return render_template('teacher.html')

# --- BACKGROUND TASK: CHECK FOR DISCONNECTS ---
def check_disconnects():
    """Checks every 3 seconds if students have gone offline."""
    while True:
        socketio.sleep(3) # Use socketio.sleep instead of time.sleep
        current_time = datetime.now()
        has_changes = False
        summary_list = []

        # Check status of all students
        for s_id, info in students_db.items():
            time_diff = (current_time - info['last_seen']).total_seconds()
            new_status = "ONLINE" if time_diff < 10 else "OFFLINE"
            
            # (Optional) You can detect if status CHANGED here to log it
            
            summary_list.append({
                "student_id": s_id,
                "name": info['name'],
                "status": new_status
            })
        
        # Always broadcast the latest status list so the UI stays fresh
        if summary_list:
            socketio.emit('class_status_update', summary_list)

# Start the background task when the server starts
socketio.start_background_task(check_disconnects)


# --- WEBSOCKET EVENTS ---
@socketio.on('get_student_data')
def handle_get_student_data(data):
    student_id = data.get('student_id')
    if student_id in students_db:
        emit('student_graph_update', {
            'student_id': student_id,
            'history': students_db[student_id]['history']
        })
    else:
        emit('student_graph_update', {'student_id': student_id, 'history': []})

@socketio.on('student_heartbeat')
def handle_heartbeat(data):
    student_id = data.get('student_id')
    if not student_id: return

    if student_id not in students_db:
        students_db[student_id] = {
            "name": f"Student {student_id}",
            "last_seen": datetime.now(),
            "history": []
        }

    now = datetime.now()
    student = students_db[student_id]
    student['last_seen'] = now

    emotion = data.get('emotion', 'Neutral')
    face_detected = data.get('face_detected', True)
    is_drowsy = data.get('is_drowsy', False)
    is_yawning = data.get('is_yawning', False)

    focus_score = calculate_focus_score(emotion, face_detected, is_drowsy, is_yawning)

    snapshot = {
        "time": now.strftime("%H:%M:%S"),
        "emotion": emotion,
        "face_detected": 1 if face_detected else 0,
        "drowsy": 1 if is_drowsy else 0,
        "yawn": 1 if is_yawning else 0,
        "focus_score": focus_score
    }

    student['history'].append(snapshot)
    if len(student['history']) > MAX_HISTORY_POINTS:
        student['history'].pop(0)

    # Send specific history update for graphs
    emit('student_graph_update', {
        'student_id': student_id,
        'history': student['history']
    }, broadcast=True)

    # --- [NEW] FORCE LIST UPDATE ---
    # This guarantees the list populates immediately without waiting for the background thread
    summary_list = []
    for s_id, info in students_db.items():
        summary_list.append({
            "student_id": s_id,
            "name": info['name'],
            "status": "ONLINE" # Assume online if we just got a heartbeat
        })
    emit('class_status_update', summary_list, broadcast=True)

if __name__ == '__main__':
    # host='0.0.0.0' is required for Docker containers to be accessible externally
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)