import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18


class ViolenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        weights = R3D_18_Weights.DEFAULT
        self.network = r3d_18(weights=weights)
        self.network.fc = nn.Linear(self.network.fc.in_features, 2)

        self.register_buffer(
            "mean",
            torch.tensor([0.43216, 0.39466, 0.37645]).view(1, 3, 1, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.22803, 0.22145, 0.21699]).view(1, 3, 1, 1, 1),
        )

    def forward(self, videos):
        # The loader returns [batch, time, channels, height, width].
        # R3D-18 expects [batch, channels, time, height, width].
        videos = videos.permute(0, 2, 1, 3, 4)
        videos = (videos - self.mean) / self.std
        return self.network(videos)
    