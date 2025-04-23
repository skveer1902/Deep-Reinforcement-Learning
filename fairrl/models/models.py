import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class QNetwork(nn.Module):
    """Deep Q-Network for decision making"""

    def __init__(self, state_size, action_size, hidden_sizes=[256, 128, 64]):
        super(QNetwork, self).__init__()

        # Create individual layer components instead of using Sequential
        # This allows us to handle batch normalization for single samples
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_size = state_size

        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            self.dropouts.append(nn.Dropout(0.2))
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, action_size)

    def forward(self, state):
        """Forward pass through the network with special handling for batch norm"""
        x = state

        # Handle batch normalization differently for single samples vs. batches
        is_single_sample = (x.dim() == 1 or x.size(0) == 1)

        if is_single_sample and x.dim() == 1:
            # Add batch dimension for single samples
            x = x.unsqueeze(0)

        for i, (layer, bn, dropout) in enumerate(zip(self.hidden_layers, self.batch_norms, self.dropouts)):
            x = layer(x)

            # Use eval mode for batch norm when processing single samples
            if is_single_sample:
                bn.eval()
                with torch.no_grad():
                    x = bn(x)
            else:
                x = bn(x)

            x = nn.functional.relu(x)
            x = dropout(x)

        x = self.output_layer(x)

        # Remove batch dimension for single samples if it was added
        if is_single_sample and state.dim() == 1:
            x = x.squeeze(0)

        return x

class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer for storing and sampling experiences with fairness awareness"""

    def __init__(self, buffer_size=20000, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000):
        """Initialize a PrioritizedReplayBuffer object.

        Args:
            buffer_size: maximum size of buffer
            alpha: how much prioritization is used (0 - no prioritization, 1 - full prioritization)
            beta_start: initial value of beta for importance-sampling weights
            beta_end: final value of beta
            beta_frames: number of frames over which to anneal beta from start to end
        """
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta_start = beta_start  # Store beta_start for annealing
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame = 1  # For beta annealing
        self.memory = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.position = 0
        self.size = 0

        # Fairness-specific attributes
        self.fairness_violations = {}  # Track experiences with fairness violations
        self.fairness_violation_count = 0

    def add(self, state, action, reward, next_state, done, fairness_info=None):
        """Add a new experience to memory with priority."""
        max_priority = self.priorities.max() if self.size > 0 else 1.0

        if len(self.memory) < self.buffer_size:
            self.memory.append((state, action, reward, next_state, done, fairness_info))
        else:
            self.memory[self.position] = (state, action, reward, next_state, done, fairness_info)

        # Set higher priority for experiences with fairness violations - balanced approach
        if fairness_info and fairness_info.get('fairness_violation', False):
            # Check violation type for better prioritization
            violation_type = fairness_info.get('violation_type')

            # Set priorities based on violation type
            if violation_type == 'demographic_parity':
                # High priority for demographic parity violations
                self.priorities[self.position] = np.float32(max_priority * 8.0)
            elif violation_type == 'equal_opportunity':
                # High priority for equal opportunity violations
                self.priorities[self.position] = np.float32(max_priority * 8.0)
            else:
                # Regular fairness violation
                self.priorities[self.position] = np.float32(max_priority * 4.0)

            # Fallback to pattern-based detection if violation_type is not available
            if violation_type is None and fairness_info.get('sensitive') is not None:
                is_female = fairness_info.get('sensitive') == 0
                is_approve = fairness_info.get('action', 0) == 1
                is_qualified = fairness_info.get('ground_truth', 0) == 1

                # Check for demographic parity specific patterns
                if (is_female and not is_approve) or (not is_female and is_approve):
                    self.priorities[self.position] = np.float32(max_priority * 8.0)
                # Check for equal opportunity specific patterns
                elif is_qualified and ((is_female and not is_approve) or (not is_female and not is_approve)):
                    self.priorities[self.position] = np.float32(max_priority * 8.0)

            self.fairness_violation_count += 1
            # Track this experience as a fairness violation
            self.fairness_violations[self.position] = True
        else:
            self.priorities[self.position] = np.float32(max_priority)
            if self.position in self.fairness_violations:
                del self.fairness_violations[self.position]

        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size, fairness_ratio=0.5):
        """Sample a batch of experiences with fairness awareness.

        Args:
            batch_size: Number of experiences to sample
            fairness_ratio: Ratio of fairness violation experiences to include
        """
        # Update beta for importance sampling weights
        self.beta = min(self.beta_end, self.beta + (self.beta_end - self.beta_start) / self.beta_frames)
        self.frame += 1

        # If we have fairness violations, ensure they're included in the sample
        if self.fairness_violations and fairness_ratio > 0:
            # Calculate how many fairness violation samples to include
            fairness_sample_size = min(int(batch_size * fairness_ratio), len(self.fairness_violations))
            regular_sample_size = batch_size - fairness_sample_size

            # Sample fairness violations
            if fairness_sample_size > 0:
                fairness_indices = random.sample(list(self.fairness_violations.keys()), fairness_sample_size)
            else:
                fairness_indices = []

            # Sample regular experiences based on priority
            if regular_sample_size > 0:
                # Create a mask to exclude fairness violation indices
                mask = np.ones(self.size, dtype=bool)
                for idx in fairness_indices:
                    if idx < self.size:
                        mask[idx] = False

                # Get priorities for non-fairness experiences
                masked_priorities = self.priorities[:self.size] * mask

                # Get valid indices (where mask is True)
                valid_indices = np.arange(self.size)[mask]
                valid_count = len(valid_indices)

                # Handle case where we have valid indices
                if valid_count > 0 and np.sum(masked_priorities) > 0:
                    # Ensure we don't try to sample more than available
                    actual_sample_size = min(regular_sample_size, valid_count)

                    # Get probabilities for just the valid indices
                    valid_priorities = masked_priorities[mask]
                    valid_probs = valid_priorities ** self.alpha
                    valid_probs = valid_probs / np.sum(valid_probs)

                    # Sample from valid indices
                    if actual_sample_size > 0:
                        sampled_indices = np.random.choice(
                            np.arange(valid_count),
                            size=actual_sample_size,
                            replace=False,
                            p=valid_probs
                        )
                        # Map back to original indices
                        regular_indices = valid_indices[sampled_indices]
                    else:
                        regular_indices = np.array([], dtype=np.int64)
                else:
                    # Fallback to random sampling if all priorities are masked or no valid indices
                    if valid_count > 0:
                        actual_sample_size = min(regular_sample_size, valid_count)
                        if actual_sample_size > 0:
                            regular_indices = np.random.choice(
                                valid_indices,
                                size=actual_sample_size,
                                replace=False
                            )
                        else:
                            regular_indices = np.array([], dtype=np.int64)
                    else:
                        regular_indices = np.array([], dtype=np.int64)
            else:
                regular_indices = []

            # Combine indices with safety check for empty arrays
            if len(fairness_indices) > 0 and len(regular_indices) > 0:
                indices = np.concatenate([fairness_indices, regular_indices])
            elif len(fairness_indices) > 0:
                indices = np.array(fairness_indices)
            elif len(regular_indices) > 0:
                indices = np.array(regular_indices)
            else:
                # If both are empty, we can't sample
                return None
        else:
            # Standard prioritized sampling if no fairness violations or fairness_ratio is 0
            if self.size == 0:
                return None

            # Calculate sampling probabilities with safety checks
            probs = self.priorities[:self.size] ** self.alpha
            probs_sum = np.sum(probs)
            if probs_sum > 0:
                probs = probs / probs_sum
            else:
                # If all priorities are zero, use uniform distribution
                probs = np.ones(self.size) / self.size

            # Ensure batch_size doesn't exceed available samples
            actual_batch_size = min(batch_size, self.size)
            indices = np.random.choice(np.arange(self.size), size=actual_batch_size, replace=False, p=probs)

        # Calculate importance sampling weights
        try:
            # Create a local copy of probs for just the selected indices
            if 'probs' in locals() and len(indices) > 0:
                # If we have a probability distribution already defined
                selected_probs = np.zeros(len(indices), dtype=np.float32)
                for i, idx in enumerate(indices):
                    if idx < len(probs):
                        selected_probs[i] = probs[idx]
                    else:
                        selected_probs[i] = 1.0 / len(indices)
                # Normalize to ensure it sums to 1
                if np.sum(selected_probs) > 0:
                    selected_probs = selected_probs / np.sum(selected_probs)
                else:
                    selected_probs = np.ones(len(indices)) / len(indices)
            else:
                # Otherwise, create uniform weights
                selected_probs = np.ones(len(indices)) / len(indices)

            # Calculate importance sampling weights
            weights = (self.size * selected_probs) ** (-self.beta)
            # Handle potential NaN or inf values
            weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
            # Normalize weights, with safety check for zero max
            max_weight = np.max(weights)
            if max_weight > 0:
                weights = weights / max_weight
            else:
                weights = np.ones_like(weights)
        except Exception as e:
            # Fallback to uniform weights if there's any error
            print(f"Warning: Error calculating importance weights: {e}. Using uniform weights.")
            weights = np.ones(len(indices), dtype=np.float32)

        # Extract experiences
        states = np.vstack([self.memory[idx][0] for idx in indices])
        actions = np.vstack([self.memory[idx][1] for idx in indices])
        rewards = np.vstack([self.memory[idx][2] for idx in indices])
        next_states = np.vstack([self.memory[idx][3] for idx in indices])
        dones = np.vstack([self.memory[idx][4] for idx in indices]).astype(np.uint8)

        # Convert to torch tensors
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float()
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones).float()
        weights = torch.from_numpy(weights).float()

        return (states, actions, rewards, next_states, dones, weights, indices)

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        for idx, priority in zip(indices, priorities):
            # Ensure priority is positive
            priority = max(priority, 1e-5)

            # Keep higher priority for fairness violations
            if idx in self.fairness_violations:
                priority = max(priority, self.priorities.max() * 2.0)

            self.priorities[idx] = priority

    def __len__(self):
        return self.size


class DDQNAgent:
    """Double Deep Q-Network agent for fair decision making with fairness-aware replay buffer"""

    def __init__(self, state_size, action_size,
                 learning_rate=0.0003,  # Further reduced learning rate for stability
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_min=0.05,  # Increased min epsilon for more exploration
                 epsilon_decay=0.998,  # Slower decay for more exploration
                 buffer_size=50000,  # Much larger buffer for more stable learning
                 batch_size=256,  # Larger batch size for more stable gradients
                 update_every=4,
                 fairness_aware=True):  # Enable fairness-aware replay buffer

        self.state_size = state_size
        self.action_size = action_size
        self.fairness_aware = fairness_aware

        # Learning parameters
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon_start  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_every = update_every

        # Q-Networks (online and target) with larger hidden layers
        self.q_network = QNetwork(state_size, action_size, hidden_sizes=[512, 256, 128])
        self.target_network = QNetwork(state_size, action_size, hidden_sizes=[512, 256, 128])
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay

        # Fairness-aware prioritized replay buffer
        if fairness_aware:
            self.memory = PrioritizedReplayBuffer(buffer_size=buffer_size)
        else:
            # Standard replay buffer
            self.memory = deque(maxlen=buffer_size)

        # Initialize time step (for updating every update_every steps)
        self.t_step = 0

        # Track fairness violations for curriculum learning
        self.fairness_violations = 0
        self.total_steps = 0
        self.fairness_ratio = 0.4  # Start with 40% fairness samples for better balance

    def step(self, state, action, reward, next_state, done, fairness_info=None):
        """Store experience in replay memory and learn if it's time"""
        self.total_steps += 1

        # Track fairness violations with focus on demographic parity
        if fairness_info and fairness_info.get('fairness_violation', False):
            self.fairness_violations += 1

            # Add action and sensitive info to fairness_info for better prioritization
            if fairness_info and 'action' not in fairness_info:
                fairness_info['action'] = action

        # Add experience to memory
        if self.fairness_aware:
            self.memory.add(state, action, reward, next_state, done, fairness_info)
        else:
            self.memory.append((state, action, reward, next_state, done))

        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # For fairness-aware buffer, ensure we have enough samples
            if self.fairness_aware and len(self.memory) > self.batch_size:
                # Gradually increase fairness ratio as training progresses
                if self.total_steps % 1000 == 0 and self.fairness_ratio < 0.8:
                    self.fairness_ratio += 0.05

                # Ensure we don't go below a minimum fairness ratio
                if self.fairness_ratio < 0.4:
                    self.fairness_ratio = 0.4

                experiences = self.memory.sample(self.batch_size, fairness_ratio=self.fairness_ratio)
                if experiences is not None:
                    self._learn_prioritized(experiences)
            # For standard buffer, use regular learning
            elif not self.fairness_aware and len(self.memory) > self.batch_size:
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
        """Randomly sample a batch of experiences from memory (for standard replay buffer)"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def _learn(self, experiences):
        """Update value parameters using batch of experience tuples (for standard replay buffer)"""
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
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self._update_target_network()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _learn_prioritized(self, experiences):
        """Update value parameters using prioritized experiences (for fairness-aware replay buffer)"""
        states, actions, rewards, next_states, dones, weights, indices = experiences

        # Get max predicted Q values for next states from target model
        # Double DQN: Use online network to select the best action
        next_action_indices = self.q_network(next_states).detach().argmax(1).unsqueeze(1)
        # Use target network to evaluate the action
        q_targets_next = self.target_network(next_states).gather(1, next_action_indices)

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.q_network(states).gather(1, actions)

        # Compute TD errors for updating priorities
        td_errors = torch.abs(q_targets - q_expected).detach().cpu().numpy()

        # Compute weighted loss (importance sampling)
        loss = (weights * nn.functional.mse_loss(q_expected, q_targets, reduction='none')).mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities in the replay buffer
        self.memory.update_priorities(indices, td_errors + 1e-5)  # Small constant for stability

        # Update target network
        self._update_target_network()

        # Update epsilon with a slower decay for fairness-aware learning
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _update_target_network(self):
        """Soft update of the target network"""
        tau = 0.01  # Further increased soft update parameter for faster target network updates
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_fairness_stats(self):
        """Get statistics about fairness violations"""
        if self.fairness_aware:
            return {
                'fairness_violations': self.fairness_violations,
                'total_steps': self.total_steps,
                'violation_ratio': self.fairness_violations / max(1, self.total_steps),
                'fairness_sample_ratio': self.fairness_ratio,
                'buffer_size': len(self.memory),
                'buffer_violation_count': self.memory.fairness_violation_count if hasattr(self.memory, 'fairness_violation_count') else 0
            }
        else:
            return {
                'fairness_violations': self.fairness_violations,
                'total_steps': self.total_steps,
                'violation_ratio': self.fairness_violations / max(1, self.total_steps),
                'buffer_size': len(self.memory)
            }