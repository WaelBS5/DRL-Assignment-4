import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import random
from collections import deque

# In order to know whether to work on cpu or gpu 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Actor Network
#Here, I'll define the deterministic policy function
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=2.0):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.max_action

    def get_action(self, state, noise=0.0, deterministic=True):

        #I'm converting the state to tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.forward(state).cpu().numpy()[0]
            if not deterministic:
                action = action + np.random.normal(0, noise, size=action.shape)
                action = np.clip(action, -self.max_action, self.max_action)
        return action

# Critic Network
#The Critic learns to approximate the action-value function. 
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1

# Replay Buffer
#The Buffer will basically store past transitions in order to experience replay. 
class ReplayBuffer:
    def __init__(self, buffer_size=1_000_000, batch_size=128): #I made a mistake of starting with 64 batch size, 128 worked well
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states = np.vstack([x[0] for x in batch])
        actions = np.vstack([x[1] for x in batch])
        rewards = np.vstack([x[2] for x in batch])
        next_states = np.vstack([x[3] for x in batch])
        dones = np.vstack([x[4] for x in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action=2.0):
        self.gamma = 0.99
        self.tau = 0.001 #tried with 0.005 but then decided to lower to 0.001 soft update rate
        self.batch_size = 128
        self.noise_std = 0.2

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        #Main & target networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)

        #Here i'll copy the initial weights to targets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        #Optimizers
        #Initial lr values
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-2)

        #Shared replay buffer
        self.buffer = ReplayBuffer(buffer_size=1_000_000, batch_size=self.batch_size)

    def select_action(self, state, add_noise=True):
        return self.actor.get_action(state, noise=self.noise_std if add_noise else 0.0, deterministic=not add_noise)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample()

        #Convert all to Tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        #Compute Target Q values: 
        next_actions = self.actor_target(next_states)
        next_q = self.critic_target(next_states, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * next_q

        #Critic Loss
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #Actor Loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #Soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename)

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename))
        self.actor.eval()

# Training function
def train_ddpg(env_name='Pendulum-v1', num_episodes=500, max_steps=200, save_path='ddpg_pendulum_model.pt'):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = DDPGAgent(state_dim, action_dim, max_action)

    best_reward = -float('inf')

    # Replay buffer with random actions
    state, _ = env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        clipped_reward = np.clip(reward, -10, 0)
        agent.buffer.add(state, action.reshape(1, -1), clipped_reward, next_state, float(done))
        state = next_state if not done else env.reset()[0]

    for episode in tqdm(range(1, num_episodes + 1)):
        state, _ = env.reset()
        episode_reward = 0

        # I was adivsed to try reseting the noise every 50 episodes and then do sort of checkpoints
        if episode % 50 == 0:
            agent.noise_std = 0.2

        for t in range(max_steps):
            action = agent.select_action(state, add_noise=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            clipped_reward = np.clip(reward, -10, 0)
            agent.buffer.add(state, action.reshape(1, -1), clipped_reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            agent.update()
            if done:
                break

        if episode % 10 == 0:
            eval_rewards = []
            for _ in range(5):
                eval_state, _ = env.reset()
                eval_reward = 0
                for _ in range(max_steps):
                    eval_action = agent.select_action(eval_state, add_noise=False)
                    eval_next_state, eval_r, term, trunc, _ = env.step(eval_action)
                    eval_state = eval_next_state
                    eval_reward += eval_r
                    if term or trunc:
                        break
                eval_rewards.append(eval_reward)

            avg_reward = np.mean(eval_rewards)
            std_reward = np.std(eval_rewards)
            score = avg_reward - std_reward
            print(f"Episode {episode}, Eval Avg: {avg_reward:.2f}, Score: {score:.2f}")

            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(save_path)
                print(f"Saved model with reward: {best_reward:.2f}")

    env.close()
    return agent

if __name__ == "__main__":
    train_ddpg()
