import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os, sys, random
from collections import deque


#I'll get dmc from parent directory 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dmc import make_dmc_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#I initially used simple Gaussian noise, scores were bad so I switched to OU noise because it helped stabilize exploration in Cartpole
#My goal is to introduce a more temporally correlated exploration of the action space with Ornstein-Uhlenbeck Noise

#Reference : openai/baselines/blob/master/baselines/ddpg/noise.py

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

#Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256) #I got low scores so I added LayerNorm to stabilize the learning
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action
        self._init_weights()

    def _init_weights(self):
        #Weight initialization helped avoid the vanishing gradient issue I faced
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return torch.tanh(self.fc3(x)) * self.max_action

    def get_action(self, state, noise=0.0, deterministic=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.forward(state).cpu().numpy()[0]
        if not deterministic:
            action += noise
        return np.clip(action, -self.max_action, self.max_action)

#Critic Network
# The critic estimates Q-values. I went with twin Q-networks (Q1 and Q2) like in TD3 to reduce overestimation bias.
# I originally implemented a single critic but noticed instability, so I introduced twin critics after studying TD3
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        #First Q network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        #Second Q network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        return self.q1(torch.cat([state, action], 1))

#Replay Buffer
#Same like in Q1
class ReplayBuffer:
    def __init__(self, size=1_000_000):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward.reshape(-1,1), next_state, done.reshape(-1,1)

    def __len__(self):
        return len(self.buffer)

#DDPGAgent
#I went through multiple iterations on this class due to training instability and low eval scores.
# My tweaks to get more than the baseline:
#LayerNorm in actor
#Twin critics with min(Q1, Q2)
#Policy delay every 2 steps (inspired by TD3)
#Soft updates with tau = 0.005 (initially I tried 0.001, but it slowed down convergence)
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.replay_buffer = ReplayBuffer()
        self.ou_noise = OUNoise(action_dim)

        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_delay = 2
        self.total_it = 0

    def select_action(self, state, noise_scale=0.1):
        action = self.actor.get_action(state, deterministic=False)
        return np.clip(action + self.ou_noise.sample() * noise_scale, -self.max_action, self.max_action)

    def update(self, batch_size=256):
        self.total_it += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        #Target noise added to next_action to simulate more conservative exploration
        with torch.no_grad():
            noise = torch.FloatTensor(np.random.normal(0, 0.2, size=action.shape)).to(device).clamp(-0.5, 0.5)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * torch.min(target_Q1, target_Q2)

        #Critic Update
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #Delayed Actor Update
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            #Soft update targets
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

#Training Loop
#Took a lot of time to tune batch size, policy delay, and add evaluation frequency manually
def train_ddpg(env_name='cartpole-balance', episodes=1000, max_steps=1000, batch_size=256, save_path='ddpg_cartpole_model.pt'):
    #DMC environment
    env = make_dmc_env(env_name, seed=np.random.randint(1e6), flatten=True, use_pixels=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    print(f"State dim: {state_dim}, Action dim: {action_dim}, Max action: {max_action}")
    agent = DDPGAgent(state_dim, action_dim, max_action)

    #To track the best model. 
    best_score = -np.inf
    for episode in tqdm(range(1, episodes+1)):
        state, _ = env.reset()
        agent.ou_noise.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = agent.select_action(state, noise_scale=0.1)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.add(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)
            if done:
                break

        # Eval : same logic like before. 
        if episode % 10 == 0:
            rewards = []
            for _ in range(5):
                s, _ = env.reset()
                ep_r = 0
                for _ in range(max_steps):
                    a = agent.actor.get_action(s, deterministic=True)
                    s, r, d, t, _ = env.step(a)
                    ep_r += r
                    if d or t: break
                rewards.append(ep_r)
            avg, std = np.mean(rewards), np.std(rewards)
            score = avg - std
            print(f"Episode {episode}, Eval Avg: {avg:.2f}, Score: {score:.2f}")
            if score > best_score:
                best_score = score
                agent.save(save_path)
                print(f"Saved model with score: {best_score:.2f}")

    print(f"Training complete. Best Score: {best_score:.2f}")
    env.close()

if __name__ == "__main__":
    train_ddpg()
