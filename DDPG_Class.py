# ddpg_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Neural Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, amax=100):
        super(Actor, self).__init__()
        self.amax = amax
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.network(state) * self.amax

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=500000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)

# Ornstein-Uhlenbeck Noise
class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.05, sigma=0.1, dt=0.01):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.ones(self.action_dim) * self.mu
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(len(x))
        self.state = x + dx
        return self.state
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

# State normalization
def normalize_state(state, state_0):
    r, lambda_, r_dot, lambda_dot = state
    r_0, lambda_0, r_dot_0, lambda_dot_0 = state_0
    
    return np.array([
        r / r_0,
        lambda_ / lambda_0,
        r_dot / r_dot_0,
        lambda_dot / lambda_dot_0
    ])

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        # Networks
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers with L2 regularization
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3, weight_decay=6e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=6e-3)
        
        # Replay buffer and noise
        self.replay_buffer = ReplayBuffer()
        self.noise = OUNoise(action_dim)
        
        # Hyperparameters from paper
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.1
        self.gradient_clip = 1.0
        
    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        
        if add_noise:
            noise = self.noise.sample()
            action += noise
            action = np.clip(action, -100, 100)
        
        return action
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Train Critic
        with torch.no_grad():
            target_next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, target_next_action)
            target_value = reward + (1 - done) * self.gamma * target_q
        
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        
        # Train Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), actor_loss.item()
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)