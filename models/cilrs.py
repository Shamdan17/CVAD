import torch.nn as nn
from torchvision.models import resnet18
import torch


class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""

    def __init__(self, freeze_backbone=False):
        super(CILRS, self).__init__()

        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            self.backbone._requires_grad = False

        self.dropout = nn.Dropout(0.5)

        self.speed_encoder = nn.Sequential(
            nn.Linear(1, 128),
            self.dropout,
            nn.ReLU(),
            nn.Linear(128, 128),
            self.dropout,
            nn.ReLU(),
            nn.Linear(128, 128),
            self.dropout,
        )

        self.speed_decoder = nn.Sequential(
            nn.Linear(512, 256),
            self.dropout,
            nn.ReLU(),
            nn.Linear(256, 256),
            self.dropout,
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.join_input = nn.Sequential(
            nn.Linear(128 + 512, 512),
            nn.ReLU(),
        )

        self.conditional_output_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512, 256),
                    self.dropout,
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    self.dropout,
                    nn.ReLU(),
                    nn.Linear(256, 3),
                    nn.Sigmoid(),
                )
                for i in range(4)
            ]
        )

    def forward(self, img, speeds, commands):
        # Set backbone to eval mode
        self.backbone.eval()
        # Encode the images
        with torch.no_grad():
            img_features = self.backbone(img)

        # Encode the speeds
        speed_features = self.speed_encoder(speeds)

        # Join the features
        joined_features = self.join_input(
            torch.cat([img_features, speed_features], dim=1)
        )

        # Predict outputs from all heads
        outputs = []
        for head in self.conditional_output_heads:
            outputs.append(
                head(joined_features).unsqueeze(-1)
            )  # Should be batch_size x 2

        # Concat the outputs at the last dimension
        outputs = torch.cat(outputs, dim=-1)  # Should be batch_size x 2 x 4

        # Output shape: batch_size x 2 x 4

        # Select the outputs for the given commands
        action_outputs = outputs[
            torch.arange(outputs.shape[0]), :, commands.reshape(-1)
        ].squeeze(
            -1
        )  # Should be batch_size x 2

        # Decode the speed from image features
        decoded_speed = self.speed_decoder(img_features)

        return action_outputs, decoded_speed
