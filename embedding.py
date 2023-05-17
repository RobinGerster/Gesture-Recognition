import torch
import cv2
import os
from tqdm import tqdm
import torchvision.transforms as transforms

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
print("Using GPU: "+ str(torch.cuda.is_available()))

def extract_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet standard deviation
        )
    ])
    while True:
        success, image = vidcap.read()
        if not success:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = transform(image)
        frames.append(frame)
    return frames  # return list of tensors directly

def create_frame_data(load_path, save_path):
    files = os.listdir(load_path)
    video_files = [f for f in files if f.endswith('.mp4')]
    print("Extracting Frames")
    for video_file in tqdm(video_files):
        video_path = os.path.join(load_path, video_file)
        frames = extract_frames(video_path)
        frames_torch = torch.stack(frames)  # stack frames into a 4D tensor

        name_without_extension = os.path.splitext(video_file)[0]
        output_filename = f"{name_without_extension}_frames.pt"
        with open(os.path.join(save_path, output_filename), 'wb') as f:
            torch.save(frames_torch, f)  # save tensor directly

def create_embeddings(load_path, save_path):
    files = os.listdir(load_path)
    pt_files = [f for f in files if f.endswith('.pt')]
    print("\nCreating embeddings")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2_vits14.to(device)
    for pt_file in pt_files:
        pt_path = os.path.join(load_path, pt_file)
        frames_torch = torch.load(pt_path)  # load tensor directly

        embeddings = []
        for frame_torch in tqdm(frames_torch):
            frame_torch = frame_torch.to(device).unsqueeze(0)
            with torch.no_grad():
                embedding = dinov2_vits14(frame_torch)[0]
            embeddings.append(embedding.cpu())
        embeddings_torch = torch.stack(embeddings)  # stack embeddings into a 2D tensor

        name_without_extension = os.path.splitext(pt_file)[0]
        output_filename = f"{name_without_extension}_embeddings.pt"
        print(embeddings_torch.shape)
        with open(os.path.join(save_path, output_filename), 'wb') as f:
            torch.save(embeddings_torch, f)  # save tensor directly

# Uncomment these lines to run the functions
#create_frame_data("./data/orig_data/train", "./data/frame_data/train")
create_frame_data("./data/orig_data/test", "./data/frame_data/test")

#create_embeddings("./data/frame_data/train", "./data/embedding_data/train")
create_embeddings("./data/frame_data/test", "./data/embedding_data/test")
