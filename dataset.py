from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ViolenceVideoDataset(Dataset):
    def __init__(
        self,
        split: str,
        root: str = "Dataset",
        frames_per_video: int = 16,
        image_size: tuple[int, int] = (112, 112),
    ):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.root = Path(root) / split
        self.frames_per_video = frames_per_video
        self.image_size = image_size
        self.samples = []

        for class_name, label in (("NonFight", 0), ("Fight", 1)):
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Folder not found: {class_dir}")

            for video_path in sorted(class_dir.glob("*.avi")):
                self.samples.append((video_path, label))

        if not self.samples:
            raise RuntimeError(f"No AVI videos found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, label = self.samples[index]
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Video has no readable frames: {video_path}")

        frame_indices = np.linspace(
            0, frame_count - 1, self.frames_per_video, dtype=int
        )

        frames = []
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()

            if not ok:
                cap.release()
                raise RuntimeError(
                    f"Could not read frame {frame_index} from {video_path}"
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.image_size)
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255
            frames.append(frame_tensor)

        cap.release()

        # Returns frames shaped [time, channels, height, width], plus the label.
        return torch.stack(frames), torch.tensor(label, dtype=torch.long)