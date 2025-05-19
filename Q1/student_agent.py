import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# I used PPO at first, but it didn't give me good scores so I had to do DDPG

#I'll define the Actor as a subclass of PyTorch Neural Network 
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=2.0):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)  
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action 
        #max_action basically to scale the output 

    #Forward Pass    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.max_action
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.forward(state).cpu().numpy()[0]
        return action

class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.state_dim = 3
        self.action_dim = 1
        self.max_action = 2.0
        #Policy Network 
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action)

        #I'll load the pre trained model and if it fails, fallback to random
        #If something breaks, the agent will return a random action, I want to avoid crashing here 
        #In my fallback system, i'm basically trying to prioritize robustness 
        try:
            self.actor.load_state_dict(torch.load('./ddpg_pendulum_model.pt'))
            self.actor.eval()
            print("DDPG model loaded successfully!")
        except Exception as e:
            print("Could not load model, using random policy")
            print(f"Error: {e}")

    #Based on the observation, we'll return the best action or random fallback
    def act(self, observation):
        try:
            return self.actor.get_action(observation)
        
        except Exception as e:
            print(f" Fallback to random due to error: {e}")
            return self.action_space.sample()
