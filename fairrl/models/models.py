import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class QNetwork(nn.Module):
    """Deep Q-Network for decision making"""
    
    def __init__(self, state_size, action_size, hidden_sizes=[128, 64]):
        super(QNetwork, self).__init__()
        
        # Build a sequential model with variable hidden layers
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass through the network"""
        return self.model(state)

class DDQNAgent:
    """Double Deep Q-Network agent for fair decision making"""
    
    def __init__(self, state_size, action_size, 
                 learning_rate=0.001, 
                 gamma=0.99, 
                 epsilon_start=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 buffer_size=10000,
                 batch_size=64,
                 update_every=4):
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Learning parameters
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon_start  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_every = update_every
        
        # Q-Networks (online and target)
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = deque(maxlen=buffer_size)
        
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Store experience in replay memory and learn if it's time"""
        # Add experience to memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self._sample_experiences()
            self._learn(experiences)
    
    def act(self, state, training=True):
        """Select an action given current state"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            return random.choice(np.arange(self.action_size))
        
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()
        
        return np.argmax(action_values.cpu().data.numpy())
    
    def _sample_experiences(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()
        
        return (states, actions, rewards, next_states, dones)
    
    def _learn(self, experiences):
        """Update value parameters using batch of experience tuples"""
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values for next states from target model
        # Double DQN: Use online network to select the best action
        next_action_indices = self.q_network(next_states).detach().argmax(1).unsqueeze(1)
        # Use target network to evaluate the action
        q_targets_next = self.target_network(next_states).gather(1, next_action_indices)
        
        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        q_expected = self.q_network(states).gather(1, actions)
        
        # Compute loss
        loss = nn.functional.mse_loss(q_expected, q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self._update_target_network()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _update_target_network(self):
        """Soft update of the target network"""
        tau = 0.001  # Soft update parameter
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)