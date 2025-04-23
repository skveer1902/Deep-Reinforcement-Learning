import numpy as np
import gym
from gym import spaces

class LoanEnvironment(gym.Env):
    """
    Reinforcement Learning environment for decision-making on the Adult dataset.
    This environment simulates a loan approval scenario where:
    - State: Features of an applicant (from the Adult dataset)
    - Action: Approve (1) or deny (0) the loan
    - Reward: Positive for correct decisions, negative for incorrect ones,
              with fairness adjustments for biased decisions
    """

    def __init__(self, X, y, sensitive, fairness_penalty=0.0, remove_sensitive=True):
        super(LoanEnvironment, self).__init__()

        self.X = X  # Features (could be a numpy array after preprocessing)

        # Convert target to numpy array for consistent indexing
        if hasattr(y, 'values'):
            self.y = y.values  # Convert pandas Series to numpy array
        else:
            self.y = y

        # Convert sensitive attribute to numpy array to ensure consistent indexing
        if hasattr(sensitive, 'values'):
            self.sensitive = sensitive.values  # Convert pandas Series to numpy array
        else:
            self.sensitive = sensitive

        self.fairness_penalty = fairness_penalty  # Penalty term for unfair decisions
        self.remove_sensitive = remove_sensitive  # Whether to remove sensitive attribute from state

        # Find the sensitive attribute index (assuming it's the last feature)
        self.sensitive_attr_idx = X.shape[1] - 1

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # Approve (1) or deny (0)

        # For sparse matrices, use shape[1] to get feature dimension
        if hasattr(X, 'toarray'):
            feature_dim = X.shape[1]
            if self.remove_sensitive:
                feature_dim -= 1  # Remove sensitive attribute from state
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32
            )
        elif isinstance(X, np.ndarray):
            feature_dim = X.shape[1]
            if self.remove_sensitive:
                feature_dim -= 1  # Remove sensitive attribute from state
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32
            )
        else:
            # Handle case when X is a DataFrame
            feature_dim = X.shape[1]
            if self.remove_sensitive:
                feature_dim -= 1  # Remove sensitive attribute from state
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32
            )

        self.reset()

    def reset(self):
        """Reset the environment to start a new episode"""
        # Choose a random applicant
        # Use shape[0] instead of len() for sparse matrices
        self.current_idx = np.random.randint(0, self.X.shape[0])

        # Convert sparse matrix row to dense array if needed
        if hasattr(self.X, 'toarray'):
            self.current_state_full = self.X[self.current_idx].toarray().flatten()
        else:
            self.current_state_full = self.X[self.current_idx]

        # Remove sensitive attribute from state if required
        if self.remove_sensitive:
            # Create a copy of the state without the sensitive attribute
            self.current_state = np.delete(self.current_state_full, self.sensitive_attr_idx)
        else:
            self.current_state = self.current_state_full

        self.current_sensitive = self.sensitive[self.current_idx]
        self.done = False
        return self.current_state

    def step(self, action):
        """
        Take action in the environment

        Args:
            action (int): 0 for deny, 1 for approve

        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Get ground truth for current applicant
        ground_truth = self.y[self.current_idx]

        # Base reward calculation
        if (action == 1 and ground_truth == 1) or (action == 0 and ground_truth == 0):
            # Correct decision
            reward = 1.0
        else:
            # Incorrect decision
            reward = -1.0

        # Add fairness component to reward (simplified version)
        if self.fairness_penalty > 0:
            if self.current_sensitive == 0:  # Female
                # If female and qualified but denied, apply fairness penalty
                if ground_truth == 1 and action == 0:
                    reward -= self.fairness_penalty
            else:  # Male
                # If male and not qualified but approved, apply fairness penalty
                if ground_truth == 0 and action == 1:
                    reward -= self.fairness_penalty

        # Episode ends after one decision
        self.done = True

        # Prepare info dict for monitoring
        info = {
            'ground_truth': ground_truth,
            'sensitive': self.current_sensitive,
            'correct_decision': action == ground_truth
        }

        # Return current state without sensitive attribute if required
        # (sparse matrices already handled in reset)
        return self.current_state, reward, self.done, info

    def render(self, mode='human'):
        """Render the environment (optional)"""
        pass