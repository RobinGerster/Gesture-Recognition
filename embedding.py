import torch
import cv2
import os
from tqdm import tqdm
import numpy as np

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
print("Using GPU: "+ str(torch.cuda.is_available()))

def extract_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    images = []
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        images.append(image)
        success, image = vidcap.read()
    return images

def create_frame_data(load_path, save_path):
    files = os.listdir(load_path)
    video_files = [f for f in files if f.endswith('.mp4')]
    print("Extracting Frames")
    for video_file in tqdm(video_files):
        video_path = os.path.join(load_path, video_file)
        images = extract_frames(video_path)
        images_np = np.array(images)

        name_without_extension = os.path.splitext(video_file)[0]
        output_filename = f"{name_without_extension}_images.npz"
        with open(os.path.join(save_path, output_filename), 'wb') as f:
            np.savez_compressed(f, images_np)

def create_embeddings(load_path, save_path):
    files = os.listdir(load_path)
    npz_files = [f for f in files if f.endswith('.npz')]
    print("\nCreating embeddings")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CUDA if available
    dinov2_vits14.to(device)  # Move model to device
    for npz_file in npz_files:
        npz_path = os.path.join(load_path, npz_file)
        images_np = np.load(npz_path)['arr_0']  # Load images from .npz file
        images_np = images_np / 255.0  # Normalize to [0, 1] range
        images_np = np.transpose(images_np, (0, 3, 1, 2))  # Change to (N, C, H, W) format

        embeddings = []
        for image_np in tqdm(images_np):
            image_torch = torch.from_numpy(image_np).float().to(device).unsqueeze(0)  # Convert to torch tensor, add batch dimension, and move to device
            with torch.no_grad():  # No need for gradients
                embedding = dinov2_vits14(image_torch)[0]  # Get embedding
            embeddings.append(embedding.cpu().numpy())  # Move to CPU and convert to numpy
        embeddings_np = np.array(embeddings)

        name_without_extension = os.path.splitext(npz_file)[0]
        output_filename = f"{name_without_extension}_embeddings.npz"
        with open(os.path.join(save_path, output_filename), 'wb') as f:
            np.savez_compressed(f, embeddings_np)



create_frame_data("./data/orig_data/train", "./data/frame_data/train")
create_frame_data("./data/orig_data/test", "./data/frame_data/test")

create_embeddings("./data/frame_data/train", "./data/embedding_data/train")
create_embeddings("./data/frame_data/test", "./data/embedding_data/test")


