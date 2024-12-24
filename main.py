import logging
import mimetypes
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import threading
import os
import opencv_pytho as cv2
from stegano import lsb
import torch
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from torchvision import models
import ffmpeg
import numpy as np

# Global Variables
input_file = None
output_file = None

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
if getattr(sys, "frozen", False):
    # Add the ffmpeg path for the bundled executable
    base_path = sys._MEIPASS
    os.environ["PATH"] += os.pathsep + os.path.join(base_path, "ffmpeg")
else:
    # Add ffmpeg path for normal script execution
    os.environ["PATH"] += os.pathsep + os.path.abspath("ffmpeg")


logging.basicConfig(level=logging.DEBUG)

def log_message(message):
    logging.debug(message)
    log_textbox.insert(tk.END, f"{message}\n")
    log_textbox.see(tk.END)

# Core Functions
def load_media(file_path):
    media_type = get_media_type(file_path)
    if media_type == "image":
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Failed to load image from {file_path}")
        return "image", [image]
    elif media_type == "video":
        cap = cv2.VideoCapture(file_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                raise ValueError(f"Failed to load a valid frame from {file_path}")
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError("No valid frames found in video")
        return "video", frames
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def save_media(media_type, frames, output_path, fps=30):
    if media_type == "image":
        cv2.imwrite(output_path, frames[0])  # Save first frame for image
    elif media_type == "video":
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()

def compress_media(input_media, output_media, media_type, crf=23):
    if media_type == "video":
        ffmpeg.input(input_media).output(
            output_media, vcodec='libx265', crf=crf, acodec='aac', preset='medium'
        ).run(overwrite_output=True)
    elif media_type == "image":
        ffmpeg.input(input_media).output(
            output_media, qscale=2
        ).run(overwrite_output=True)

def process_media(input_path, output_path):
    global output_file
    media_type, frames = load_media(input_path)
    processed_frames = []

    log_message(f"Loaded {media_type}: {len(frames)} frames")
    log_message("Loaded ResNet50 model for adversarial perturbation")

    model = models.resnet50(pretrained=True)
    model.eval()

    for i, frame in enumerate(frames):
        log_message(f"Processing frame {i+1}/{len(frames)}")

        noisy_frame = apply_noise(frame)
        stego_frame = embed_message(noisy_frame, f"Protected Frame {i+1}")
        adversarial_frame = generate_adversarial_example(stego_frame, model)
        processed_frames.append(adversarial_frame)

    save_media(media_type, processed_frames, output_path)
    compressed_output_path = f"compressed_{os.path.basename(output_path)}"
    compress_media(output_path, compressed_output_path, media_type)

    output_file = compressed_output_path


def get_media_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        if "image" in mime_type:
            return "image"
        elif "video" in mime_type:
            return "video"
    raise ValueError(f"Unsupported file type for: {file_path}")


# Supporting Functions
def apply_noise(image, noise_level=0.01):
    noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def embed_message(frame, message, temp_path="temp_frame.png"):
    temp_path_with_extension = os.path.splitext(temp_path)[0] + ".png"
    cv2.imwrite(temp_path_with_extension, frame)
    secret_image = lsb.hide(temp_path_with_extension, message)
    secret_image.save(temp_path_with_extension)
    return cv2.imread(temp_path_with_extension)

def generate_adversarial_example(image, model):
    image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 255),
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters()),
        input_shape=(3, image.shape[0], image.shape[1]),
        nb_classes=1000,
    )
    attack = FastGradientMethod(estimator=classifier, eps=0.01)
    adversarial_image = attack.generate(x=image_tensor.numpy())
    return adversarial_image[0].transpose(1, 2, 0).astype(np.uint8)

# Logging for GUI
def log_message(message):
    log_textbox.insert(tk.END, f"{message}\n")
    log_textbox.see(tk.END)

# GUI Functions
def select_file():
    global input_file
    input_file = filedialog.askopenfilename(
        filetypes=[("Media Files", "*.jpg *.png *.jpeg *.bmp *.tiff *.mp4 *.avi *.mkv *.mov")]
    )
    if input_file:
        input_label.configure(text=f"Selected: {os.path.basename(input_file)}")

def process_file():
    global input_file, output_file
    if not input_file:
        messagebox.showerror("Error", "Please select a file to process.")
        return

    output_path = os.path.splitext(input_file)[0] + "_processed.mp4"

    # Start processing in a separate thread to avoid blocking the GUI
    threading.Thread(target=process_and_finish, args=(input_file, output_path), daemon=True).start()

def process_and_finish(input_path, output_path):
    try:
        log_message(f"Starting processing: {input_path}")
        process_media(input_path, output_path)
        log_message("Processing completed successfully!")
        log_message(f"Output saved at: {output_file}")
    except Exception as e:
        log_message(f"Error during processing: {str(e)}")

# GUI Setup
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("AI-Powered Media Processor")
app.geometry("600x400")

# UI Components
title_label = ctk.CTkLabel(app, text="AI-Powered Media Processor", font=("Arial", 18))
title_label.pack(pady=10)

input_label = ctk.CTkLabel(app, text="No file selected", font=("Arial", 14))
input_label.pack(pady=5)

select_button = ctk.CTkButton(app, text="Select File", command=select_file)
select_button.pack(pady=10)

process_button = ctk.CTkButton(app, text="Process File", command=process_file)
process_button.pack(pady=10)

log_textbox = tk.Text(app, height=10, width=80, wrap=tk.WORD)
log_textbox.pack(pady=10)

# Start GUI
app.mainloop()
