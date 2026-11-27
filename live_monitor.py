from collections import deque
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
import torch
from model import ViolenceClassifier


FRAMES_PER_WINDOW = 16
WINDOW_SECONDS = 5
INFERENCE_INTERVAL = 1.0
FIGHT_THRESHOLD = 0.40
ALERT_AFTER_WINDOWS = 2
ALERT_COOLDOWN_SECONDS = 15
IMAGE_SIZE = (112, 112)
ALARM_SOUND = "/System/Library/Sounds/Basso.aiff"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = ViolenceClassifier().to(device)
model.load_state_dict(
    torch.load("checkpoints/fine_tuned_model.pth", map_location=device)
)
model.eval()

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError(
        "Could not open the camera. Check that VS Code has camera permission "
        "in System Settings > Privacy & Security > Camera."
    )

frame_buffer = deque()
last_inference_time = 0.0
last_alert_time = -ALERT_COOLDOWN_SECONDS
fight_score = None
consecutive_high_scores = 0
alert_latched = False

print("Camera monitor started. Press Q in the video window to quit.")

try:
    while True:
        ok, frame = camera.read()
        if not ok:
            print("Could not read from the camera.")
            break

        now = time.monotonic()

        small = cv2.resize(frame, IMAGE_SIZE)
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        frame_buffer.append((now, small))

        while frame_buffer and now - frame_buffer[0][0] > WINDOW_SECONDS:
            frame_buffer.popleft()

        buffer_ready = (
            len(frame_buffer) >= FRAMES_PER_WINDOW
            and frame_buffer[-1][0] - frame_buffer[0][0] >= WINDOW_SECONDS * 0.8
        )

        if buffer_ready and now - last_inference_time >= INFERENCE_INTERVAL:
            indices = np.linspace(
                0, len(frame_buffer) - 1, FRAMES_PER_WINDOW, dtype=int
            )
            sampled_frames = np.stack(
                [frame_buffer[index][1] for index in indices]
            )

            video = (
                torch.from_numpy(sampled_frames)
                .permute(0, 3, 1, 2)
                .float()
                .div(255)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                scores = torch.softmax(model(video), dim=1)[0]
                fight_score = scores[1].item()

            last_inference_time = now

            if fight_score >= FIGHT_THRESHOLD:
                consecutive_high_scores += 1
            else:
                consecutive_high_scores = 0
                alert_latched = False

            cooldown_finished = (
                now - last_alert_time >= ALERT_COOLDOWN_SECONDS
            )

            if (
                consecutive_high_scores >= ALERT_AFTER_WINDOWS
                and not alert_latched
                and cooldown_finished
            ):
                if Path(ALARM_SOUND).is_file():
                    subprocess.Popen(
                        [
                            "sh",
                            "-c",
                            'for i in 1 2 3; do afplay "$1"; done',
                            "alarm",
                            ALARM_SOUND,
                        ]
                    )
                else:
                    print("\a", end="", flush=True)

                print(
                    f"ALERT: possible violence "
                    f"(fight score {fight_score:.1%})"
                )
                last_alert_time = now
                alert_latched = True

        if fight_score is None:
            status = "Building 5-second video window..."
        elif fight_score >= FIGHT_THRESHOLD:
            status = f"Possible violence ({fight_score:.1%})"
        else:
            status = f"Fight score: {fight_score:.1%}"

        color = (0, 0, 255) if (
            fight_score is not None and fight_score >= FIGHT_THRESHOLD
        ) else (0, 255, 0)

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        cv2.imshow("Violence Monitor - press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()