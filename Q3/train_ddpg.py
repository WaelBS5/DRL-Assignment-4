import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import sys
import random
from collections import deque

#make dmc env wrapper : 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dmc import make_dmc_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#I used a simple deque-based buffer with a cap of 1M transitions
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards).reshape(-1, 1),
            np.array(next_states),
            np.array(dones).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    #I initially tried without normalization, but I was getting poor scores
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=400):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.l3 = nn.Linear(hidden_dim // 2, action_dim)
        self.max_action = max_action
        #Xavier initialization
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.xavier_uniform_(self.l3.weight)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x)) * self.max_action

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return self.forward(state).detach().cpu().numpy()[0]


#Same as in Q1 and Q2 
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim + action_dim, hidden_dim // 2)
        self.l3 = nn.Linear(hidden_dim // 2, 1)
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.xavier_uniform_(self.l3.weight)

    def forward(self, state, action):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(torch.cat([x, action], dim=1)))
        return self.l3(x)

class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=400):
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=2.5e-4)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=2.5e-4)

        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.9992)
        self.critic_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.9992)

        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 128
        self.noise = OUNoise(action_dim, sigma=0.2)
        self.max_action = max_action

    def select_action(self, state, add_noise=True):
        self.actor.eval()
        action = self.actor.get_action(state)
        self.actor.train()
        if add_noise:
            action = np.clip(action + self.noise.sample(), -self.max_action, self.max_action)
        return action

    def update(self, replay_buffer):
        state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        target_action = self.actor_target(next_state)
        target_q = self.critic_target(next_state, target_action)
        target_q = reward + (1 - done) * self.gamma * target_q

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor_scheduler.step()
        self.critic_scheduler.step()

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        torch.save({"actor": self.actor.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor.eval()

def train_ddpg(env_name="humanoid-walk", num_episodes=10000, max_steps=1000, save_path="ddpg_humanoid_model.pt", checkpoint_dir="checkpoints"):
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Max action: {max_action}")

    agent = DDPGAgent(state_dim, action_dim, max_action)
    os.makedirs(checkpoint_dir, exist_ok=True)
    replay_buffer = ReplayBuffer(1_000_000)

    best_reward = -np.inf
    min_replay_size = 5000

    state, _ = env.reset()
    print("Filling replay buffer")
    for _ in tqdm(range(min_replay_size), desc="Replay buffer warm-up"):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, terminated or truncated)
        state = next_state if not (terminated or truncated) else env.reset()[0]

    for episode in tqdm(range(1, num_episodes + 1)):
        state, _ = env.reset()
        agent.noise.reset()
        episode_reward = 0

        for t in range(max_steps):
            action = agent.select_action(state, add_noise=(episode < num_episodes // 2))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > agent.batch_size:
                agent.update(replay_buffer)

            if done:
                break

        if episode % 10 == 0:
            rewards = []
            for _ in range(3):
                state, _ = env.reset()
                total_reward = 0
                while True:
                    action = agent.select_action(state, add_noise=False)
                    state, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        break
                rewards.append(total_reward)
            avg_reward = np.mean(rewards)
            print(f"Episode {episode}, Eval Avg: {avg_reward:.2f}")

            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(save_path)
                print(f"Best model saved with reward: {best_reward:.2f}")

        if episode % 500 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"ddpg_checkpoint_{episode}.pt")
            agent.save(checkpoint_path)
            print(f"Checkpoint saved at episode {episode}")

    env.close()
    agent.save(save_path)
    print(f"Final model saved with best reward: {best_reward:.2f}")

if __name__ == "__main__":
    train_ddpg()
