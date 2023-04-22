from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import json
import torch
import os


class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""

    def __init__(self, data_root, train=True):
        self.data_root = data_root
        # Your code here
        self.num_digits = 8
        self.train = train

        if self.train:
            # If training, we use augmentations that retain the same affordances
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    # transforms.ColorJitter(
                    #     brightness=0.2, contrast=0.2, saturation=0.2
                    # ),
                    transforms.ToTensor(),
                    # transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                    transforms.Normalize((0.406, 0.456, 0.485), (0.225, 0.224, 0.229)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.406, 0.456, 0.485), (0.225, 0.224, 0.229)),
                ]
            )

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        # Your code here
        prefix = str(index).zfill(self.num_digits)

        rgb_path = self.data_root + "/rgb/" + prefix + ".png"

        rgb = Image.open(rgb_path)

        rgb_tensor = self.transform(rgb).flip(0)  # -> BGR -> RGB

        measurements_path = self.data_root + "/measurements/" + prefix + ".json"
        measurements = json.load(open(measurements_path))

        # Example
        # {"speed": 0.0, "throttle": 0.75, "brake": 0.0, "steer": 0.022938258945941925, "command": 3, "route_dist": 0.0, "route_angle": 0.014603931028367375, "lane_dist": 0.7769609093666077, "lane_angle": 0.11314811706542968, "hazard": false, "hazard_dist": 25, "tl_state": 0, "tl_dist": 45.0, "is_junction": false}

        # Make into a tensor
        # [
        #     measurements["speed"],
        #     measurements["throttle"],
        #     measurements["brake"],
        #     measurements["steer"],
        #     measurements["command"],
        #     measurements["route_dist"],
        #     measurements["route_angle"],
        #     measurements["lane_dist"],
        #     measurements["lane_angle"],
        #     measurements["hazard"],
        #     measurements["hazard_dist"],
        #     measurements["tl_state"],
        #     measurements["tl_dist"],
        #     measurements["is_junction"],
        # ]

        measurements["acceleration"] = measurements["throttle"] - measurements["brake"]

        # measurements["action"] = torch.tensor(
        #     [
        #         normalize(measurements["acceleration"], 0.3, 0.35),
        #         normalize(measurements["steer"], 0.0, 0.15),
        #     ]
        # )

        measurements["action"] = torch.tensor(
            [
                measurements["throttle"],
                measurements["brake"],
                measurements["steer"],
            ]
        )

        measurements["cnt_affordances"] = torch.tensor(
            [
                normalize(measurements["lane_dist"], 0.15, 0.25),
                normalize(measurements["route_angle"], 0.0, 0.05),
                normalize(measurements["tl_dist"], 25, 20),
                # measurements["lane_dist"],
                # measurements["route_angle"],
                # measurements["tl_dist"],
            ]
        )

        measurements["bin_affordances"] = torch.tensor(
            [
                measurements["tl_state"],
            ]
        )

        measurements["speed"] = normalize(measurements["speed"], 3.0, 2.25)

        # Make everything in the measurements dict a tensor
        for key in measurements:
            if not isinstance(measurements[key], torch.Tensor):
                measurements[key] = torch.tensor([measurements[key]])

        return rgb_tensor, measurements

    def __len__(self):
        return len(os.listdir(self.data_root + "/rgb/"))
        # return 1000


def normalize(x, mean, std):
    """Normalize a tensor with mean and standard deviation."""
    return (x - mean) / std
