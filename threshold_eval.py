import torch
from torch.utils.data import DataLoader
from dataset import ViolenceVideoDataset
from model import ViolenceClassifier


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

loader = DataLoader(
    ViolenceVideoDataset("val"),
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

model = ViolenceClassifier().to(device)
model.load_state_dict(
    torch.load("checkpoints/fine_tuned_model.pth", map_location=device)
)
model.eval()

all_labels = []
all_fight_scores = []

with torch.no_grad():
    for videos, labels in loader:
        scores = torch.softmax(model(videos.to(device)), dim=1)[:, 1]
        all_labels.extend(labels.tolist())
        all_fight_scores.extend(scores.cpu().tolist())

labels = torch.tensor(all_labels)
fight_scores = torch.tensor(all_fight_scores)

for threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
    predictions = (fight_scores >= threshold).long()

    true_fights = labels == 1
    actual_nonfights = labels == 0
    detected_fights = predictions == 1
    detected_nonfights = predictions == 0

    tp = (true_fights & detected_fights).sum().item()
    fn = (true_fights & detected_nonfights).sum().item()
    fp = (actual_nonfights & detected_fights).sum().item()
    tn = (actual_nonfights & detected_nonfights).sum().item()

    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0

    print(
        f"Threshold {threshold:.2f} | "
        f"fight recall: {recall:.1%} ({fn} missed) | "
        f"fight precision: {precision:.1%} | "
        f"false alarms: {fp}/{fp + tn}"
    )