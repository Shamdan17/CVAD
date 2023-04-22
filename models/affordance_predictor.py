import torch.nn as nn
import torch
from torchvision.models import resnet18


class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""

    def __init__(self, freeze_backbone=False):
        super(AffordancePredictor, self).__init__()

        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        if freeze_backbone:
            self.backbone._requires_grad = False

        self.dropout = nn.Dropout(0.6)
        self.freeze_backbone = freeze_backbone

        self.command_embedding = nn.Embedding(4, 32)

        self.join_input = nn.Sequential(
            nn.Linear(32 + 512, 512),
        )

        self.conditional_output = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 256),
            self.dropout,
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        self.unconditional_output = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 256),
            self.dropout,
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, img, commands):
        # Encode the images
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                img_features = self.backbone(img)
        else:
            img_features = self.backbone(img)

        # Encode the commands
        command_features = self.command_embedding(commands.squeeze(1))

        # Join the features
        joined_features = self.join_input(
            torch.cat([command_features, img_features], dim=1)
        )

        # Predict the affordances
        cond_affordances = self.conditional_output(joined_features)
        uncond_affordances = self.unconditional_output(img_features)

        return torch.cat((cond_affordances, uncond_affordances), dim=1)
