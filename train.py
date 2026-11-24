from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import ViolenceVideoDataset
from model import ViolenceClassifier

BATCH_SIZE = 2
EPOCHS = 3

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

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

# Start from the best model saved during the first training run.
model.load_state_dict(
    torch.load("checkpoints/best_model.pth", map_location=device)
)

# Freeze the network, then fine-tune its last layer and classifier.
for parameter in model.network.parameters():
    parameter.requires_grad = False

for parameter in model.network.layer4.parameters():
    parameter.requires_grad = True

for parameter in model.network.fc.parameters():
    parameter.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    [
        {"params": model.network.layer4.parameters(), "lr": 0.00001},
        {"params": model.network.fc.parameters(), "lr": 0.0001},
    ]
)

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dir / "fine_tuned_model.pth"
best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    print(f"\nStarting epoch {epoch + 1}/{EPOCHS}...", flush=True)

    model.train()
    model.network.eval()  # Keep batch-normalization statistics fixed.

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_index, (videos, labels) in enumerate(train_loader, start=1):
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

        if batch_index % 50 == 0:
            print(
                f"  Training batch {batch_index}/{len(train_loader)}",
                flush=True,
            )

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
        f"val accuracy: {val_accuracy:.1%}",
        flush=True,
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved best fine-tuned model to {checkpoint_path}.", flush=True)

print(f"\nBest fine-tuned validation accuracy: {best_val_accuracy:.1%}")