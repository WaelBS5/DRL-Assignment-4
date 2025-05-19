import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

# Actor model
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=400):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, action_dim)
        self.max_action = max_action

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.max_action

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.forward(state).cpu().numpy()[0]
        return action

# Main agent
class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), dtype=np.float64)
        self.state_dim = 67
        self.action_dim = 21
        self.max_action = 1.0

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action)

        try:
            model_path = os.path.join(os.path.dirname(__file__), "ddpg_humanoid_model.pt")
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            self.actor.load_state_dict(checkpoint["actor"])
            self.actor.eval()
            print("DDPG model loaded successfully!")
        except Exception as e:
            print("Could not load model, using random policy")
            print(f"Error: {e}")

    def act(self, observation):
        try:
            return self.actor.get_action(observation)
        except Exception as e:
            print(f"Fallback to random due to error: {e}")
            return self.action_space.sample()
