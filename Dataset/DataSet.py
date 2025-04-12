import os
import cv2
import torch
import torchvision


class CustomVideoDataset(torch.utils.data.Dataset):
    def __init__(self, videos_dir: str, transform: torchvision.transforms.Compose = None):
        super(CustomVideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.transform = transform
        # Define classes mapping
        self.classes = {"Red card": 0, "scoring": 1, "tackling": 2}
        self.video_files = []
        self.labels = []

        # Iterate over class directories and add valid video file paths
        for class_dir in os.listdir(self.videos_dir):
            class_path = os.path.join(self.videos_dir, class_dir)
            if os.path.isdir(class_path):
                label = self.classes.get(class_dir)
                if label is None:
                    continue
                for video in os.listdir(class_path):
                    video_path = os.path.join(class_path, video)
                    if os.path.isfile(video_path):
                        self.video_files.append(video_path)
                        self.labels.append(label)

    def __len__(self) -> int:
        return len(self.video_files)

    def _load_video(self, video_path: str) -> list:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        return frames

    def __getitem__(self, idx: int) -> dict:
        video_path = self.video_files[idx]
        label = self.labels[idx]
        frames = self._load_video(video_path)
        video_tensor = torch.stack(frames)  # Shape: [T, C, H, W]
        label_tensor = torch.tensor(label, dtype=torch.long)
        return {"video": video_tensor, "label": label_tensor}
    @staticmethod
    def variable_length_collate(batch):
        videos = [item["video"] for item in batch]
        labels = torch.stack([item["label"] for item in batch])
        return {"video": videos, "label": labels}

"""
transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])
dataset = CustomVideoDataset(videos_dir="/media/muhammad/New Volume/dataset/football match action video dataset",
                             transform=transform)
video_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=CustomVideoDataset.variable_length_collate)

for batch in video_loader:
    input_data, labels = batch["video"], batch["label"]
    print("Batch labels:", labels)
    for i, video in enumerate(input_data):
        print(f"Video {i} shape: {video.shape}")
    break
"""