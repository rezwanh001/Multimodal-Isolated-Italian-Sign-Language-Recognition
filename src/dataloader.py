import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
import numpy as np

class SessioniDataset(Dataset):
    def __init__(self, root_dir, split='train', num_frames=16, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.num_frames = num_frames
        self.transform = transform
        self.samples = []
        self.labels = []
        
        if split == 'train':
            # Load training samples with class labels
            for class_folder in os.listdir(root_dir):
                class_id = int(class_folder.split('_')[0])
                class_path = os.path.join(root_dir, class_folder)
                for sample_folder in os.listdir(class_path):
                    sample_path = os.path.join(class_path, sample_folder)
                    sample_id = int(sample_folder.split('_')[1])
                    self.samples.append((sample_id, sample_path))
                    self.labels.append(class_id)
        else:
            # Load val/test samples (no class labels)
            for sample_folder in os.listdir(root_dir):
                sample_id = int(sample_folder.split('_')[1])
                sample_path = os.path.join(root_dir, sample_folder)
                self.samples.append((sample_id, sample_path))
                self.labels.append(-1)  # Placeholder for val/test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id, sample_path = self.samples[idx]
        label = self.labels[idx]

        # Load RGB video
        rgb_path = os.path.join(sample_path, f'SAMPLE_{sample_id}_RGB.mkv')
        rgb_frames = self.load_video(rgb_path, self.num_frames)

        # Load RDM videos
        rdm1_path = os.path.join(sample_path, f'SAMPLE_{sample_id}_RDM1.mp4')
        rdm2_path = os.path.join(sample_path, f'SAMPLE_{sample_id}_RDM2.mp4')
        rdm3_path = os.path.join(sample_path, f'SAMPLE_{sample_id}_RDM3.mp4')
        rdm1_frames = self.load_video(rdm1_path, self.num_frames)
        rdm2_frames = self.load_video(rdm2_path, self.num_frames)
        rdm3_frames = self.load_video(rdm3_path, self.num_frames)

        # Apply transforms
        if self.transform:
            rgb_frames = torch.stack([self.transform(frame) for frame in rgb_frames])
            rdm1_frames = torch.stack([self.transform(frame) for frame in rdm1_frames])
            rdm2_frames = torch.stack([self.transform(frame) for frame in rdm2_frames])
            rdm3_frames = torch.stack([self.transform(frame) for frame in rdm3_frames])

        # Stack RDM frames (treat as channels or process separately)
        rdm_frames = torch.stack([rdm1_frames, rdm2_frames, rdm3_frames], dim=0)  # Shape: [3, T, C, H, W]

        return {
            'sample_id': sample_id,
            'rgb': rgb_frames,  # Shape: [T, C, H, W]
            'rdm': rdm_frames,  # Shape: [3, T, C, H, W]
            'label': torch.tensor(label, dtype=torch.long)
        }

    def load_video(self, video_path, num_frames):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            raise ValueError(f"Empty video: {video_path}")
        
        # Uniformly sample frames
        step = max(1, frame_count // num_frames)
        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # [C, H, W]
                frames.append(frame)
            if len(frames) >= num_frames:
                break
        cap.release()
        
        # Pad if fewer frames
        while len(frames) < num_frames:
            frames.append(frames[-1])
        return frames[:num_frames]

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])