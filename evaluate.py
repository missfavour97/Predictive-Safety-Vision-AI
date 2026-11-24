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

# Rows are actual labels; columns are predicted labels.
# Class 0 = NonFight, class 1 = Fight.
matrix = torch.zeros((2, 2), dtype=torch.long)

with torch.no_grad():
    for videos, labels in loader:
        predictions = model(videos.to(device)).argmax(dim=1).cpu()
        labels = labels.cpu()

        for actual, predicted in zip(labels, predictions):
            matrix[actual, predicted] += 1

tn, fp = matrix[0, 0].item(), matrix[0, 1].item()
fn, tp = matrix[1, 0].item(), matrix[1, 1].item()

fight_recall = tp / (tp + fn) if tp + fn else 0
fight_precision = tp / (tp + fp) if tp + fp else 0

print("Confusion matrix (rows = actual, columns = predicted)")
print("                 Pred NonFight   Pred Fight")
print(f"Actual NonFight       {tn:4}          {fp:4}")
print(f"Actual Fight          {fn:4}          {tp:4}")
print(f"Fight recall:    {fight_recall:.1%}  (actual fights detected)")
print(f"Fight precision: {fight_precision:.1%}  (fight alerts that are correct)")