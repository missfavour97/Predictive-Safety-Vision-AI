from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import ViolenceVideoDataset
from model import ViolenceClassifier


BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 0.001

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

train_loader = DataLoader(
    ViolenceVideoDataset("train"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    ViolenceVideoDataset("val"),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

model = ViolenceClassifier().to(device)

# Train only the final classifier layer for this first baseline.
for parameter in model.network.parameters():
    parameter.requires_grad = False
for parameter in model.network.fc.parameters():
    parameter.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.network.fc.parameters(),
    lr=LEARNING_RATE,
)

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()
    model.network.eval()  # Keep the frozen network's batch-normalization layers fixed.

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for videos, labels in train_loader:
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(videos)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        train_correct += (logits.argmax(dim=1) == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            labels = labels.to(device)

            logits = model(videos)
            loss = criterion(logits, labels)

            val_loss += loss.item() * labels.size(0)
            val_correct += (logits.argmax(dim=1) == labels).sum().item()
            val_total += labels.size(0)

    train_accuracy = train_correct / train_total
    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"train loss: {train_loss / train_total:.4f} | "
        f"train accuracy: {train_accuracy:.1%} | "
        f"val loss: {val_loss / val_total:.4f} | "
        f"val accuracy: {val_accuracy:.1%}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
        print("Saved best model.")