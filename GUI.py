import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
import threading
import time
from PIL import Image, ImageTk

# ---------------------------------------
# IMPORTANT RASPBERRY PI CAMERA NOTES:
# ---------------------------------------
# If you are using the official Raspberry Pi Camera Module,
# be sure to enable the camera interface in 'raspi-config' and load
# the V4L2 driver by running:
#     sudo modprobe bcm2835-v4l2
# This ensures that cv2.VideoCapture(0) will work correctly.
#
# For USB webcams, this code should work as long as the camera is at /dev/video0.

# ----------------------------
# Define your class names
# ----------------------------
CLASSES_LIST = ['LumbarSideBends', 'QuadrupedThoracicRotation', 'SupineNeckLift']

# ----------------------------
# Global variables for recording and live feed
# ----------------------------
recording_in_progress = False
recording_duration = 10  # seconds
recording_start_time = None
recorded_frames = []  # list to hold frames when recording

live_cap = cv2.VideoCapture(0)
if not live_cap.isOpened():
    print("Error: Unable to open camera for live feed.")
else:
    live_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    live_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# ----------------------------
# Optical Flow Functions (Lucas-Kanade)
# ----------------------------
def compute_optical_flow(prev_frame, next_frame):
    """
    Compute optical flow using the Lucas-Kanade method between two frames.
    Returns:
        good_old: Array of points from the previous frame.
        good_new: Array of corresponding points from the next frame.
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    feature_params = dict(maxCorners=3000, qualityLevel=0.01, minDistance=0.1)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    if p0 is None:
        return [], []
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)
    if p1 is not None and st is not None:
        st = st.flatten()
        good_old = p0[st == 1]
        good_new = p1[st == 1]
    else:
        good_old, good_new = [], []
    return good_old, good_new

def create_mei_mhi(flow, shape, tau=10):
    """
    Create Motion Energy Image (MEI) and Motion History Image (MHI) from optical flow vectors.
    """
    mei = np.zeros(shape, dtype=np.float32)
    mhi = np.zeros(shape, dtype=np.float32)
    
    for (new, old) in zip(flow[1], flow[0]):
        x1, y1 = new.ravel()
        x2, y2 = old.ravel()
        cv2.line(mei, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
        cv2.line(mhi, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
    
    # Decrease MHI values over time (simple demonstration).
    mhi[mhi > 0] -= 255 / tau  
    mhi[mhi < 0] = 0
    
    return mei, mhi

def compute_optical_flow_sequence_lk(frames, target_size=(224, 224), tau=10):
    """
    For a list of frames, resize them to target_size (width, height),
    compute Lucas-Kanade optical flow between consecutive frames, and obtain the MEI.
    Returns an array of shape (num_frames, target_height, target_width, 1).
    """
    resized_frames = [cv2.resize(frame, target_size) for frame in frames]
    flow_sequence = []
    for i in range(len(resized_frames) - 1):
        prev_frame = resized_frames[i]
        next_frame = resized_frames[i + 1]
        good_old, good_new = compute_optical_flow(prev_frame, next_frame)
        if len(good_old) == 0 or len(good_new) == 0:
            # If no features found, just use a zero frame
            mei = np.zeros((target_size[1], target_size[0]), dtype=np.float32)
        else:
            mei, _ = create_mei_mhi((good_old, good_new), (target_size[1], target_size[0]), tau=tau)
            mei = mei / 255.0  # Normalize to [0,1]
        mei = np.expand_dims(mei, axis=-1)  # Add channel dimension
        flow_sequence.append(mei.astype(np.float32))
    
    # Pad the sequence if necessary to match the number of frames.
    while len(flow_sequence) < len(frames):
        zero_frame = np.zeros((target_size[1], target_size[0], 1), dtype=np.float32)
        flow_sequence.append(zero_frame)
    
    return np.array(flow_sequence)

# ----------------------------
# RGB Preprocessing Function
# ----------------------------
def preprocess_rgb_frames(frames, target_size=(96, 96)):
    """
    Resize each frame to target_size (width, height), convert from BGR to RGB,
    and normalize pixel values to [0,1]. Returns an array of shape (num_frames, target_height, target_width, 3).
    """
    rgb_frames = []
    for frame in frames:
        resized = cv2.resize(frame, target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        rgb_frames.append(rgb)
    return np.array(rgb_frames)

# ----------------------------
# Frame Sampling Function
# ----------------------------
def sample_frames_from_video(video_path, num_frames):
    """
    Samples `num_frames` uniformly from the input video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        print("Warning: Video has fewer frames than requested; using available frames.")
        num_frames = total_frames
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    current_index = 0
    ret = True
    while ret and current_index < total_frames:
        ret, frame = cap.read()
        if current_index in indices and frame is not None:
            frames.append(frame)
        current_index += 1
    cap.release()
    return frames

# ----------------------------
# Ensure Video Resolution Function
# ----------------------------
def ensure_video_resolution(frames, target_size=(320, 240)):
    """
    Resize each frame to the specified target resolution (width, height).
    """
    return [cv2.resize(frame, target_size) for frame in frames]

# ----------------------------
# Inference Function
# ----------------------------
def run_inference_on_video(video_path, model, num_frames=30):
    """
    Samples frames from the video, ensures resolution, preprocesses for both streams,
    and runs inference using the provided model.
    Returns:
        pred_class (int or None): predicted class index
        predictions (ndarray or None): raw prediction probabilities
        flow_sequence (ndarray): the computed optical-flow sequence for display
    """
    frames = sample_frames_from_video(video_path, num_frames)
    if len(frames) < num_frames or len(frames) == 0:
        print("Error: Not enough frames in the video for inference.")
        return None, None, None

    # Ensure each frame is 320x240
    frames = ensure_video_resolution(frames, target_size=(320, 240))

    # Preprocess for RGB
    rgb_sequence = preprocess_rgb_frames(frames, target_size=(96, 96))

    # Preprocess for optical flow
    flow_sequence = compute_optical_flow_sequence_lk(frames, target_size=(224, 224), tau=10)
    
    # Add batch dimension to each stream
    rgb_input = np.expand_dims(rgb_sequence, axis=0)
    flow_input = np.expand_dims(flow_sequence, axis=0)
    
    # Run model prediction
    predictions = model.predict([rgb_input, flow_input])
    pred_class = np.argmax(predictions, axis=1)[0]
    return pred_class, predictions, flow_sequence

# ----------------------------
# Display Optical Flow in Tkinter
# ----------------------------
def display_optical_flow_sequence(flow_sequence, index=0):
    """
    Iterates through the given flow_sequence array and displays each frame
    in the 'optical_flow_label' via Tkinter, non-blocking (using root.after).
    """
    if flow_sequence is None or len(flow_sequence) == 0:
        return

    if index < len(flow_sequence):
        flow_img = (flow_sequence[index][:, :, 0] * 255).astype(np.uint8)
        color_flow = cv2.applyColorMap(flow_img, cv2.COLORMAP_BONE)
        
        # Convert to PIL Image and then to ImageTk
        pil_img = Image.fromarray(color_flow)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        
        optical_flow_label.imgtk = imgtk  # keep a reference
        optical_flow_label.config(image=imgtk)
        
        # Schedule the next frame update
        root.after(100, lambda: display_optical_flow_sequence(flow_sequence, index + 1))
    else:
        # Once done, you can optionally clear or leave the last frame displayed
        pass

# ----------------------------
# Recording & Live Feed Functions
# ----------------------------
def update_live_feed():
    """
    Continuously grabs frames from the live camera, updates the Tkinter Label,
    and if recording is active, saves each frame to the recorded_frames list.
    """
    ret, frame = live_cap.read()
    if ret:
        # Convert the frame (BGR) to RGB and then to ImageTk format
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk  # Prevent garbage collection
        video_label.configure(image=imgtk)
        
        # If recording is active, save a copy of the frame
        if recording_in_progress:
            recorded_frames.append(frame.copy())
            elapsed = int(time.time() - recording_start_time)
            remaining = max(0, recording_duration - elapsed)
            timer_label.config(text=f"Recording... {remaining} seconds remaining")
            if elapsed >= recording_duration:
                stop_recording()
    # Schedule next frame update
    root.after(15, update_live_feed)

def stop_recording():
    """
    Stops the recording, writes the recorded frames to a video file,
    and then starts video inference and optical flow display.
    """
    global recording_in_progress
    recording_in_progress = False
    timer_label.config(text="Recording complete!")
    
    video_path = "recorded_video.avi"
    if recorded_frames:
        h, w, _ = recorded_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
        for f in recorded_frames:
            out.write(f)
        out.release()
        # Process the recorded video in a separate thread
        threading.Thread(target=process_recorded_video, args=(video_path,), daemon=True).start()
    # Re-enable the record button after a short delay
    root.after(100, lambda: record_button.config(state="normal"))

def process_recorded_video(video_path):
    """
    Runs inference on the recorded video and updates the status label.
    Then displays the optical flow frames in the GUI.
    """
    root.after(0, lambda: status_label.config(text="Processing video for inference..."))

    predicted_class, probs, flow_sequence = run_inference_on_video(video_path, late_fusion_model, num_frames=30)
    
    if predicted_class is not None:
        predicted_class_name = CLASSES_LIST[predicted_class]
        confidence_percentage = probs[0][predicted_class] * 100
        if confidence_percentage < 68:
            result_text = f"Predicted Class: Unknown\n"
        else:
            result_text = (
                f"Predicted Class: {predicted_class_name}\n"
                f"Confidence: {confidence_percentage:.2f}%"
            )
    else:
        result_text = "Error processing video."
    
    root.after(0, lambda: status_label.config(text=result_text))

    # Display optical flow sequence after 2 seconds
    def show_flow():
        display_optical_flow_sequence(flow_sequence, 0)
    root.after(2000, show_flow)

def pre_record_countdown(t):
    """
    Displays a countdown before starting recording.
    """
    if t >= 0:
        timer_label.config(text=f"Starting recording in {t} seconds")
        root.after(1000, lambda: pre_record_countdown(t-1))
    else:
        start_recording()

def start_recording():
    """
    Begins recording by setting the recording flag and timestamp.
    """
    global recording_in_progress, recording_start_time, recorded_frames
    recording_in_progress = True
    recording_start_time = time.time()
    recorded_frames = []

def on_record():
    """
    Triggered when the 'Record and Classify' button is pressed.
    Initiates a 5-second pre-record countdown.
    """
    record_button.config(state="disabled")
    pre_record_countdown(5)

# ----------------------------
# Load the Trained Model
# ----------------------------
model_path = r"/home/rpi/Desktop/FORDEMO_2/ForDemo2/FEB24/CNN_GRU_MODEL1.h5"
try:
    late_fusion_model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    late_fusion_model = None

# ----------------------------
# Tkinter GUI Setup (Dynamic Scaling)
# ----------------------------
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 600
root = tk.Tk()
root.title("Video Classification")
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
root.resizable(False, False)
root.configure(background="#1e1e1e")

# Define dynamic fonts and paddings (base resolution: 1024x600)
scale = WINDOW_WIDTH / 1024
FONT_TITLE = ("Helvetica", int(32 * scale), "bold")
FONT_CONTENT = ("Helvetica", int(28 * scale))
FONT_BUTTON = ("Helvetica", int(28 * scale), "bold")
FONT_STATUS = ("Helvetica", int(24 * scale))

TITLE_PADX = int(20 * scale)
TITLE_PADY = int(10 * scale)
BUTTON_PADX = int(60 * scale)
BUTTON_PADY = int(20 * scale)
CONTENT_PADX = int(40 * scale)
CONTENT_PADY = int(40 * scale)
INFO_PADX = int(20 * scale)
INFO_PADY_TOP = int(40 * scale)
INFO_PADY_BOTTOM = int(20 * scale)
exit_ipady = int(10 * scale)

# Colors
BACKGROUND_COLOR = "#1e1e1e"
TITLE_BG = "#3a3f47"
TITLE_FG = "white"
CONTENT_BG = "#282c34"
INFO_BG = "#3a3f47"
BUTTON_BG = "#61afef"
BUTTON_ACTIVE_BG = "#528dc7"
TEXT_COLOR = "white"

# Custom Title Bar
title_bar = tk.Frame(root, bg=TITLE_BG, relief="raised", bd=0)
title_bar.pack(side="top", fill="x", pady=(TITLE_PADY, 0))

title_label = tk.Label(title_bar, text="Video Classification", bg=TITLE_BG, fg=TITLE_FG, font=FONT_TITLE)
title_label.pack(side="left", padx=TITLE_PADX, pady=TITLE_PADY)

def minimize_window():
    root.iconify()

def close_window():
    if live_cap is not None:
        live_cap.release()
    root.destroy()

minimize_button = tk.Button(title_bar, text="_", command=minimize_window, bg=TITLE_BG, fg=TITLE_FG,
                            relief="flat", font=FONT_TITLE, padx=TITLE_PADX//2, pady=TITLE_PADY//2, bd=0)
minimize_button.pack(side="right", padx=TITLE_PADX)

close_button = tk.Button(title_bar, text="X", command=close_window, bg=TITLE_BG, fg=TITLE_FG,
                         relief="flat", font=FONT_TITLE, padx=TITLE_PADX//2, pady=TITLE_PADY//2, bd=0)
close_button.pack(side="right", padx=TITLE_PADX)

def start_move(event):
    root.x = event.x
    root.y = event.y

def on_move(event):
    deltax = event.x - root.x
    deltay = event.y - root.y
    x = root.winfo_x() + deltax
    y = root.winfo_y() + deltay
    root.geometry(f"+{x}+{y}")

title_bar.bind("<Button-1>", start_move)
title_bar.bind("<B1-Motion>", on_move)

# -------------------------------------------------------
# Main Content Split: Left (Video Feed) and Right (Status)
# -------------------------------------------------------
content_frame = tk.Frame(root, bg=CONTENT_BG)
content_frame.pack(fill="both", expand=True)

# Left Frame (Live feed + optical flow)
left_frame = tk.Frame(content_frame, bg=CONTENT_BG)
left_frame.pack(side="left", fill="both", expand=True)

# Right Frame (Status labels, countdown, buttons)
right_frame = tk.Frame(content_frame, bg=CONTENT_BG)
right_frame.pack(side="right", fill="both", expand=True)

# Place the live camera feed label
video_label = tk.Label(left_frame, bg=CONTENT_BG)
video_label.pack(expand=True, fill="both", padx=CONTENT_PADX, pady=CONTENT_PADY)

# Place the optical flow label (below the live feed)
optical_flow_label = tk.Label(left_frame, bg=CONTENT_BG)
optical_flow_label.pack(expand=True, fill="both", padx=CONTENT_PADX, pady=(0, CONTENT_PADY))

# Info Frame on the right
info_frame = tk.Frame(right_frame, bg=INFO_BG, bd=3, relief="ridge")
info_frame.pack(pady=(INFO_PADY_TOP, INFO_PADY_BOTTOM), padx=INFO_PADX, fill="x")

status_label = tk.Label(info_frame, text="Status: Idle", bg=INFO_BG, fg=TEXT_COLOR,
                        font=FONT_STATUS, wraplength=WINDOW_WIDTH//2, justify="center")
status_label.pack(pady=(TITLE_PADY, TITLE_PADY), padx=INFO_PADX)

timer_label = tk.Label(info_frame, text="Timer: --", bg=INFO_BG, fg=TEXT_COLOR, font=FONT_CONTENT)
timer_label.pack(pady=(0, INFO_PADY_BOTTOM), padx=INFO_PADX)

# Button Frame (record & exit) on the right
button_frame = tk.Frame(right_frame, bg=CONTENT_BG)
button_frame.pack(expand=True, fill="x", pady=(INFO_PADY_BOTTOM, INFO_PADY_BOTTOM))

record_button = tk.Button(button_frame, text="Record and Classify", command=on_record,
                          font=FONT_BUTTON, bg=BUTTON_BG, fg=TEXT_COLOR, activebackground=BUTTON_ACTIVE_BG,
                          padx=BUTTON_PADX, pady=BUTTON_PADY)
record_button.pack(fill="x", pady=(0, 20))

exit_button = tk.Button(button_frame, text="Exit", command=close_window,
                        font=FONT_BUTTON, bg="#d9534f", fg=TEXT_COLOR, activebackground="#c9302c",
                        padx=BUTTON_PADX, pady=BUTTON_PADY)
exit_button.pack(fill="x", ipady=exit_ipady)

# Start the live camera feed updates
update_live_feed()

root.mainloop()
