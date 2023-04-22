import os

import yaml

from carla_env.env import Env
import torch
from models.cilrs import CILRS

# Transforms
from torchvision import transforms


class Evaluator:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.load_agent()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize((0.406, 0.456, 0.485), (0.225, 0.224, 0.229)),
            ]
        )

    def load_agent(self):
        # Your code here
        self.agent = CILRS().eval()
        self.agent.load_state_dict(torch.load("cilrs_model.ckpt"))  # Load the weights
        self.agent.cuda()

    def generate_action(self, rgb, command, speed):
        # Your code here

        rgb = self.transform(torch.tensor(rgb).float().permute((2, 0, 1))/255).cuda().unsqueeze(0)
        command = torch.tensor([command]).unsqueeze(-1).cuda()
        speed = torch.tensor([speed]).unsqueeze(-1).cuda().float()

        action, _ = self.agent(rgb, speed, command)

        pred_throttle = action[0, 0].item()
        pred_brake = action[0, 1].item()
        pred_steer = action[0, 2].item()

        return pred_throttle, pred_steer, pred_brake

    def take_step(self, state):
        rgb = state["rgb"]
        command = state["command"]
        speed = state["speed"]
        throttle, steer, brake = self.generate_action(rgb, command, speed)
        action = {"throttle": throttle, "brake": brake, "steer": steer}
        state, reward_dict, is_terminal = self.env.step(action)
        return state, is_terminal

    def evaluate(self, num_trials=100):
        terminal_histogram = {}
        for i in range(num_trials):
            state, _, is_terminal = self.env.reset()
            for i in range(5000):
                if is_terminal:
                    break
                state, is_terminal = self.take_step(state)
            if not is_terminal:
                is_terminal = ["timeout"]
            terminal_histogram[is_terminal[0]] = (
                terminal_histogram.get(is_terminal[0], 0) + 1
            )
        print("Evaluation over. Listing termination causes:")
        for key, val in terminal_histogram.items():
            print(f"{key}: {val}/100")


def main():
    with open(os.path.join("configs", "cilrs.yaml"), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        evaluator = Evaluator(env, config)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
