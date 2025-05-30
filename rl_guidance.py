import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from environment import plot_trajectories, MissileGuidanceEnv

# Neural Networks as per paper architecture
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

# Experience Replay Buffer
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
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.1, dt=0.01):
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

# State normalization as per paper
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
        self.tau = 0.1  # Paper suggests lower for stability
        self.gradient_clip = 1.0  # ρ from paper
        
    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        
        if add_noise:
            noise = self.noise.sample()
            action += noise
            action = np.clip(action, -100, 100)  # amax constraint
        
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

# Training loop following paper's approach
def train_ddpg(env, agent, episodes=100):
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # Store initial state for normalization
        state_0 = state.copy()
        
        while not done:
            # Normalize state
            normalized_state = normalize_state(state, state_0)
            
            # Select action
            action = agent.select_action(normalized_state)
            
            # Take step
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.replay_buffer.push(normalized_state, action, reward, 
                                   normalize_state(next_state, state_0), done)
            
            # Update networks
            if len(agent.replay_buffer) >= agent.batch_size:
                critic_loss, actor_loss = agent.train()
            
            state = next_state
            episode_reward += reward
            step += 1
            
            if done:
                break
        
        rewards.append(episode_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
    
    return rewards

# Training with characteristic scenarios
initial_conditions = [
    # [r, lambda, gamma_M, gamma_T, tau]
    [4000, np.radians(-10), np.radians(0), np.radians(140), 0.1],
    [4000, np.radians(10), np.radians(20), np.radians(160), 0.3],
    [6000, np.radians(-10), np.radians(0), np.radians(140), 0.1],
    [6000, np.radians(10), np.radians(20), np.radians(160), 0.3],
    # ... add all scenarios from Table 2
]

# Initialize agent
agent = DDPGAgent(state_dim=4, action_dim=1)

# Train on fixed scenarios first
print("Training on fixed characteristic scenarios...")
for i, ic in enumerate(initial_conditions):
    env = MissileGuidanceEnv(logging=False)
    # Set initial conditions manually
    rewards = train_ddpg(env, agent, episodes=100)
    print(f"Scenario {i+1} completed. Best reward: {max(rewards):.2f}")

# Train with random initialization
print("Training with random initialization...")
env = MissileGuidanceEnv(logging=False)
final_rewards = train_ddpg(env, agent, episodes=200)

# Test the trained agent
print("Testing trained agent...")
test_env = MissileGuidanceEnv(logging=True)
state = test_env.reset()
state_0 = state.copy()
done = False

while not done:
    normalized_state = normalize_state(state, state_0)
    action = agent.select_action(normalized_state, add_noise=False)
    state, reward, done, _ = test_env.step(action)

print(f"Final range: {test_env.r:.2f} m")
plot_trajectories(test_env)