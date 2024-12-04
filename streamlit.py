import logging
import os
import sys
import cv2
import ffmpeg
import numpy as np
import torch
from stegano import lsb
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from torchvision import models
import streamlit as st
from main import compress_media

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Function to load media
def load_media(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".jpg", ".png", ".jpeg", ".bmp", ".tiff"]:
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Failed to load image from {file_path}")
        return "image", [image]
    elif ext in [".mp4", ".avi", ".mkv", ".mov"]:
        cap = cv2.VideoCapture(file_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError("No valid frames found in video")
        return "video", frames
    else:
        raise ValueError("Unsupported file format. Provide an image or video.")

# Function to save media
def save_media(media_type, frames, output_path, fps=30):
    if media_type == "image":
        cv2.imwrite(output_path, frames[0])  # Save the first frame
    elif media_type == "video":
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()

# Function to process media
def process_media(input_path, output_path):
    media_type, frames = load_media(input_path)
    processed_frames = []

    st.write(f"Loaded {media_type}: {len(frames)} frames")
    st.write("Loaded ResNet50 model for adversarial perturbation")

    model = models.resnet50(pretrained=True)
    model.eval()

    for i, frame in enumerate(frames):
        st.write(f"Processing frame {i+1}/{len(frames)}")
        noisy_frame = apply_noise(frame)
        stego_frame = embed_message(noisy_frame, f"Protected Frame {i+1}")
        adversarial_frame = generate_adversarial_example(stego_frame, model)
        processed_frames.append(adversarial_frame)

    save_media(media_type, processed_frames, output_path)

    # Generate compressed output path
    output_ext = os.path.splitext(output_path)[1]
    compressed_output_path = os.path.splitext(output_path)[0] + f"_compressed{output_ext}"
    compress_media(output_path, compressed_output_path, media_type)

    return compressed_output_path

# Supporting functions
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

# Streamlit App
st.title("AI-Powered Media Processor")

uploaded_file = st.file_uploader("Upload your media file (image/video)", type=["jpg", "png", "jpeg", "bmp", "tiff", "mp4", "avi", "mkv", "mov"])

if uploaded_file:
    # Dynamic paths
    input_ext = os.path.splitext(uploaded_file.name)[1]
    temp_input_path = os.path.join("temp_input", uploaded_file.name)
    temp_output_path = os.path.splitext(temp_input_path)[0] + f"_processed{input_ext}"

    # Save uploaded file locally
    os.makedirs("temp_input", exist_ok=True)
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Process File"):
        try:
            st.write(f"Processing: {temp_input_path}")
            output_path = process_media(temp_input_path, temp_output_path)
            st.success(f"Processing completed! Download your file below.")
            st.download_button("Download Processed File", open(output_path, "rb"), file_name=os.path.basename(output_path))
        except Exception as e:
            st.error(f"Error: {e}")
