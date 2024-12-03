import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import threading
import os
import cv2
from stegano import lsb
import torch
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from torchvision import models
import ffmpeg

# Global Variables
input_file = None
output_file = None

# Core Functions
def load_media(file_path):
    """Determine if input is an image or video and load it accordingly."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".jpg", ".png", ".jpeg", ".bmp", ".tiff"]:
        # Load image
        image = cv2.imread(file_path)
        return "image", [image]
    elif ext in [".mp4", ".avi", ".mkv", ".mov"]:
        # Load video
        cap = cv2.VideoCapture(file_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return "video", frames
    else:
        raise ValueError("Unsupported file format. Provide an image or video.")

def save_media(media_type, frames, output_path, fps=30):
    """Save frames as an image or video."""
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
    """Compress the media file using FFmpeg."""
    if media_type == "video":
        ffmpeg.input(input_media).output(
            output_media, vcodec='libx265', crf=crf, acodec='aac', preset='medium'
        ).run(overwrite_output=True)
    elif media_type == "image":
        ffmpeg.input(input_media).output(
            output_media, qscale=2  # Adjust for image compression
        ).run(overwrite_output=True)

def process_media(input_path, output_path):
    """Process media with noise, metadata embedding, adversarial perturbation, and compression."""
    global output_file
    media_type, frames = load_media(input_path)
    processed_frames = []

    # Load Pretrained Model
    model = models.resnet50(pretrained=True)
    model.eval()

    # Process each frame
    for i, frame in enumerate(frames):
        # Add noise
        noisy_frame = apply_noise(frame)

        # Embed hidden metadata
        temp_image_path = "temp_frame.png"
        stego_frame = embed_message(noisy_frame, f"Protected Frame {i+1}", temp_image_path)

        # Adversarial perturbation
        adversarial_frame = generate_adversarial_example(stego_frame, model)
        processed_frames.append(adversarial_frame)

    # Save processed media
    save_media(media_type, processed_frames, output_path)
    compressed_output_path = f"compressed_{os.path.basename(output_path)}"
    compress_media(output_path, compressed_output_path, media_type)

    # Update global output file
    output_file = compressed_output_path

# Supporting Functions
def apply_noise(image, noise_level=0.01):
    """Add imperceptible noise to the image."""
    noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def embed_message(frame, message, temp_path):
    """Embed a hidden message into a frame."""
    cv2.imwrite(temp_path, frame)  # Save frame temporarily for steganography
    secret_image = lsb.hide(temp_path, message)
    secret_image.save(temp_path)
    return cv2.imread(temp_path)

def generate_adversarial_example(image, model):
    """Generate an adversarial example using FGSM."""
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
    progress_bar.start()
    threading.Thread(target=lambda: process_and_finish(input_file, output_path)).start()

def process_and_finish(input_path, output_path):
    try:
        process_media(input_path, output_path)
        progress_bar.stop()
        messagebox.showinfo("Success", f"File processed successfully!\nSaved at: {output_file}")
    except Exception as e:
        progress_bar.stop()
        messagebox.showerror("Error", str(e))

# GUI Setup
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("AI-Powered Media Processor")
app.geometry("500x300")

# UI Components
title_label = ctk.CTkLabel(app, text="AI-Powered Media Processor", font=("Arial", 18))
title_label.pack(pady=10)

input_label = ctk.CTkLabel(app, text="No file selected", font=("Arial", 14))
input_label.pack(pady=5)

select_button = ctk.CTkButton(app, text="Select File", command=select_file)
select_button.pack(pady=10)

process_button = ctk.CTkButton(app, text="Process File", command=process_file)
process_button.pack(pady=10)

progress_bar = ctk.CTkProgressBar(app, mode="indeterminate")
progress_bar.pack(pady=20, padx=50)

# Start GUI
app.mainloop()
