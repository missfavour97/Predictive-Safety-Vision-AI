import argparse
import cv2
import numpy as np
import torch
from model import ViolenceClassifier


def load_video(path, frames_per_video=16, image_size=(112, 112)):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {path}")

    indices = np.linspace(0, frame_count - 1, frames_per_video, dtype=int)
    frames = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()

        if not ok:
            cap.release()
            raise RuntimeError(f"Could not read frame {index} from {path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, image_size)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255)

    cap.release()
    return torch.stack(frames).unsqueeze(0)


parser = argparse.ArgumentParser()
parser.add_argument("video", help="Path to an AVI video")
args = parser.parse_args()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ViolenceClassifier().to(device)
model.load_state_dict(
    torch.load("checkpoints/fine_tuned_model.pth", map_location=device)
)
model.eval()

video = load_video(args.video).to(device)

with torch.no_grad():
    scores = torch.softmax(model(video), dim=1)[0]

labels = ["NonFight", "Fight"]
prediction = scores.argmax().item()

print("Prediction:", labels[prediction])
print(f"Fight score: {scores[1].item():.1%}")