import pandas as pd
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================
# 1. GUI FUNCTION (View in Window)
# ==========================================
def view_results_from_csv(tab_widget, csv_path):
    # 1. Clear previous widgets
    for widget in tab_widget.winfo_children():
        widget.destroy()

    # 2. Setup Scrollable Canvas
    canvas = tk.Canvas(tab_widget)
    scrollbar = ttk.Scrollbar(tab_widget, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Mousewheel scrolling
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    # Bind mousewheel only when hovering over canvas
    canvas.bind("<Enter>", lambda _: canvas.bind_all("<MouseWheel>", _on_mousewheel))
    canvas.bind("<Leave>", lambda _: canvas.unbind_all("<MouseWheel>"))

    # 3. Load and Process Data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        lbl = tk.Label(scrollable_frame, text=f"Error reading file: {e}", fg="red")
        lbl.pack(pady=20)
        return

    if df.empty:
        lbl = tk.Label(scrollable_frame, text="No data available yet.", fg="gray")
        lbl.pack(pady=20)
        return

    # Data Type Conversions
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Emotion'] = df['Emotion'].fillna('Neutral').astype(str)
    
    # Standardize Booleans (Handle 'Yes'/'No' strings)
    for col in ['DrowsinessAlert', 'YawnAlert', 'FaceMissing']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x) == 'Yes' else 0)

    # Focus Score Logic
    emotion_score_map = {
        'Happy': 2, 'Surprise': 2, 'Neutral': 1,
        'Sad': 0, 'Angry': -1, 'Fear': -2, 'Disgust': -3
    }

    def compute_focus_score(row):
        # Priority: FaceMissing (-1) -> Drowsy (-3) -> Yawn (-2) -> Emotion
        if row.get('FaceMissing', 0) == 1:
            return -1
        elif row.get('DrowsinessAlert', 0) == 1:
            return -3
        elif row.get('YawnAlert', 0) == 1:
            return -2
        else:
            return emotion_score_map.get(row['Emotion'], 0)

    df['FocusScore'] = df.apply(compute_focus_score, axis=1)

    # 4. Resampling Logic (Prevent graph overcrowding)
    duration = (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds()
    df = df.set_index('Timestamp')

    resample_interval = None
    if duration > 1800: resample_interval = '30s'
    elif duration > 900: resample_interval = '10s'
    elif duration > 300: resample_interval = '5s'

    if resample_interval:
        # Define how to aggregate different columns
        agg_dict = {
            'Emotion': lambda x: x.mode()[0] if not x.mode().empty else 'Neutral',
            'DrowsinessAlert': 'max',
            'YawnAlert': 'max',
            'FaceMissing': 'max',
            'FocusScore': 'mean'
        }
        # Only aggregate existing columns
        valid_agg = {k: v for k, v in agg_dict.items() if k in df.columns}
        df = df.resample(resample_interval).agg(valid_agg).dropna().reset_index()
    else:
        df = df.reset_index()

    total_score = df['FocusScore'].sum()
    avg_score = df['FocusScore'].mean()

    # ================= PLOTTING HELPERS =================
    def add_figure_to_frame(fig):
        frame = ttk.Frame(scrollable_frame)
        frame.pack(fill='both', expand=True, pady=10, padx=10)
        
        canvas_plot = FigureCanvasTkAgg(fig, master=frame)
        canvas_plot.draw()
        canvas_plot.get_tk_widget().pack(fill='both', expand=True)

    # Graph 1: Emotion Over Time
    fig1 = Figure(figsize=(10, 4), dpi=100)
    ax1 = fig1.add_subplot(111)
    ax1.plot(df['Timestamp'], df['Emotion'], marker='o', linestyle='-', alpha=0.7, markersize=3)
    ax1.set_title("Emotion Timeline")
    ax1.grid(True, linestyle='--', alpha=0.6)
    fig1.autofmt_xdate()
    add_figure_to_frame(fig1)

    # Graph 2: Face Presence
    fig2 = Figure(figsize=(10, 3), dpi=100)
    ax2 = fig2.add_subplot(111)
    # Invert FaceMissing logic for display: 1 = Face Present
    face_present = df['FaceMissing'].apply(lambda x: 0 if x==1 else 1)
    ax2.fill_between(df['Timestamp'], face_present, step="pre", alpha=0.4, color='blue')
    ax2.set_title("Face Presence (1=Present, 0=Missing)")
    ax2.set_yticks([0, 1])
    ax2.grid(True, linestyle='--', alpha=0.6)
    fig2.autofmt_xdate()
    add_figure_to_frame(fig2)

    # Graph 3: Alerts
    fig3 = Figure(figsize=(10, 4), dpi=100)
    ax3 = fig3.add_subplot(111)
    ax3.step(df['Timestamp'], df['DrowsinessAlert'], where='post', label='Drowsy', color='red')
    ax3.step(df['Timestamp'], df['YawnAlert'], where='post', label='Yawn', color='orange')
    ax3.set_title("Alert Events")
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['No', 'Yes'])
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)
    fig3.autofmt_xdate()
    add_figure_to_frame(fig3)

    # Graph 4: Focus Score
    fig4 = Figure(figsize=(10, 4), dpi=100)
    ax4 = fig4.add_subplot(111)
    ax4.plot(df['Timestamp'], df['FocusScore'], color='purple', marker='.', linestyle='-')
    ax4.axhline(0, color='black', linewidth=0.8)
    ax4.set_title(f"Focus Score (Avg: {avg_score:.2f})")
    ax4.grid(True, linestyle='--', alpha=0.6)
    fig4.autofmt_xdate()
    add_figure_to_frame(fig4)


# ==========================================
# 2. PDF GENERATION FUNCTION (Minio Upload)
# ==========================================
def generate_pdf_report(csv_path, output_pdf_path):
    """Generates a PDF report from the CSV log."""
    try:
        df = pd.read_csv(csv_path)
        
        # Data Cleaning
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Emotion'] = df['Emotion'].fillna('Neutral').astype(str)
        
        # Convert Yes/No to 1/0
        for col in ['DrowsinessAlert', 'YawnAlert', 'FaceMissing']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 1 if str(x) == 'Yes' else 0)

        # Focus Score Logic
        scores = {'Happy': 2, 'Surprise': 2, 'Neutral': 1, 'Sad': 0, 'Angry': -1, 'Fear': -2, 'Disgust': -3}
        def get_score(row):
            if row.get('FaceMissing', 0): return -1
            if row.get('DrowsinessAlert', 0): return -3
            if row.get('YawnAlert', 0): return -2
            return scores.get(row.get('Emotion'), 0)

        df['FocusScore'] = df.apply(get_score, axis=1)

        # Generate PDF
        with PdfPages(output_pdf_path) as pdf:
            # Page 1: Focus & Emotion
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(df['Timestamp'], df['FocusScore'], color='purple', label='Focus Score')
            ax1.set_title(f"Focus Score Report (Avg: {df['FocusScore'].mean():.2f})")
            ax1.set_ylabel("Score (-3 to +2)")
            ax1.grid(True)
            ax1.legend()
            fig1.autofmt_xdate()
            pdf.savefig(fig1)
            plt.close()
            
            # Page 2: Alerts
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(df['Timestamp'], df['DrowsinessAlert'], color='red', label='Drowsiness')
            ax2.plot(df['Timestamp'], df['YawnAlert'], color='orange', label='Yawn', alpha=0.7)
            ax2.set_title("Alert Timeline")
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['Safe', 'Alert'])
            ax2.legend()
            ax2.grid(True)
            fig2.autofmt_xdate()
            pdf.savefig(fig2)
            plt.close()
            
        return True
    except Exception as e:
        print(f"PDF Error: {e}")
        return False