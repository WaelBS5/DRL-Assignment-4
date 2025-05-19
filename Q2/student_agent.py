import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Same like in Q1, I tried PPO and SAC, the results were very bad so i switched to DDPG
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim) #Decided to add LayerNorm to help with training stability
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return torch.tanh(self.fc3(x)) * self.max_action

    
    def get_action(self, state, deterministic=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.forward(state).cpu().numpy()[0]
        return action

# Same logic as in Q1 :  i load the trained weights- if model not available i'll fallback to random 
class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float64)
        self.state_dim = 5  
        self.action_dim = 1
        self.max_action = 1.0

        self.actor = Actor(self.state_dim, self.action_dim, max_action=self.max_action)
        try:
            model_path = os.path.join(os.path.dirname(__file__), "ddpg_cartpole_model.pt")
            self.actor.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            self.actor.eval()
            print("DDPG model loaded successfully!")
        except Exception as e:
            print("Could not load model, using random policy")
            print(f"Error: {e}")
        
    def act(self, observation):
        try:
            return self.actor.get_action(observation, deterministic=True)
        except Exception as e:
            print(f"Fallback to random due to error: {e}")
            return self.action_space.sample()
