import torch
import numpy as np
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

class IGExplainer:
    """
    Implementation of Integrated Gradients for explaining RL model decisions
    and detecting bias in decision patterns.
    """
    
    def __init__(self, model, feature_names=None):
        """
        Initialize the explainer with a trained model
        
        Args:
            model: The trained RL model (Q-Network)
            feature_names: Names of features for visualization
        """
        self.model = model
        self.feature_names = feature_names
        self.ig = IntegratedGradients(self.model_forward)
    
    def model_forward(self, inputs):
        """Forward function for the model that returns Q-values"""
        return self.model(inputs)
    
    def explain_decision(self, state, action_idx=None):
        """
        Compute attributions for a decision using Integrated Gradients
        
        Args:
            state: The input state to explain
            action_idx: The action index to explain (if None, uses max Q-value action)
            
        Returns:
            attributions: Feature importance scores
        """
        # Handle sparse matrices by converting to dense array
        if hasattr(state, 'toarray'):
            state = state.toarray().flatten()
        
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        else:
            state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
        
        # If no action is specified, use the model's chosen action
        if action_idx is None:
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action_idx = torch.argmax(q_values, dim=1).item()
        
        # Define the target function for the specified action
        def target_fn(inputs):
            return self.model_forward(inputs)[:, action_idx]
        
        # Baseline is typically zero vector
        baseline = torch.zeros_like(state_tensor)
        
        # Compute attributions
        attributions = self.ig.attribute(state_tensor, baseline, target=action_idx)
        
        return attributions.squeeze(0).detach().numpy(), action_idx
    
    def compare_attributions(self, states_group1, states_group2, sensitive_feature_idx=None):
        """
        Compare attributions between two groups (e.g., male vs. female)
        to identify potential bias in decision patterns
        
        Args:
            states_group1: States from group 1 (e.g., female applicants)
            states_group2: States from group 2 (e.g., male applicants)
            sensitive_feature_idx: Index of the sensitive feature to highlight
            
        Returns:
            dict: Comparison metrics of attributions between groups
        """
        # Handle case when input is sparse matrix
        if hasattr(states_group1, 'toarray') and hasattr(states_group1, 'shape'):
            # Convert entire sparse matrix to dense array if it's small enough
            if states_group1.shape[0] <= 100:  # Only convert if reasonable size
                states_group1 = states_group1.toarray()
            # Otherwise process rows individually in the loop below
        
        if hasattr(states_group2, 'toarray') and hasattr(states_group2, 'shape'):
            if states_group2.shape[0] <= 100:
                states_group2 = states_group2.toarray()
        
        # Get attributions for both groups
        attributions_group1 = []
        actions_group1 = []
        
        for i in range(states_group1.shape[0]):
            # Extract individual state (handles both dense and sparse)
            if hasattr(states_group1, 'toarray') and not isinstance(states_group1, np.ndarray):
                state = states_group1[i]  # Will be handled in explain_decision
            else:
                state = states_group1[i]
                
            attr, action = self.explain_decision(state)
            attributions_group1.append(attr)
            actions_group1.append(action)
        
        attributions_group2 = []
        actions_group2 = []
        
        for i in range(states_group2.shape[0]):
            # Extract individual state
            if hasattr(states_group2, 'toarray') and not isinstance(states_group2, np.ndarray):
                state = states_group2[i]
            else:
                state = states_group2[i]
                
            attr, action = self.explain_decision(state)
            attributions_group2.append(attr)
            actions_group2.append(action)
        
                # Convert to arrays
        attributions_group1 = np.array(attributions_group1)
        attributions_group2 = np.array(attributions_group2)
        
        # Compute average attributions for each group
        avg_attr_group1 = np.mean(attributions_group1, axis=0)
        avg_attr_group2 = np.mean(attributions_group2, axis=0)
        
        # Compute attribution differences
        attr_diff = avg_attr_group1 - avg_attr_group2
        
        # Compute divergence in decision patterns
        approve_rate_group1 = np.mean([1 if a == 1 else 0 for a in actions_group1])
        approve_rate_group2 = np.mean([1 if a == 1 else 0 for a in actions_group2])
        
        # Return comparison metrics
        return {
            "avg_attributions_group1": avg_attr_group1,
            "avg_attributions_group2": avg_attr_group2,
            "attribution_differences": attr_diff,
            "approve_rate_group1": approve_rate_group1,
            "approve_rate_group2": approve_rate_group2,
            "approve_rate_difference": approve_rate_group1 - approve_rate_group2
        }
    
    def visualize_attributions(self, attributions, title="Feature Attributions"):
        """Visualize feature attributions"""
        if self.feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(attributions))]
        else:
            feature_names = self.feature_names
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if attr >= 0 else 'red' for attr in attributions]
        plt.bar(range(len(attributions)), attributions, color=colors)
        plt.xticks(range(len(attributions)), feature_names, rotation=90)
        plt.title(title)
        plt.xlabel("Features")
        plt.ylabel("Attribution Score")
        plt.tight_layout()
        return plt
    
    def visualize_attribution_comparison(self, comparison_results, title="Attribution Comparison"):
        """Visualize attribution comparison between groups"""
        avg_attr_group1 = comparison_results["avg_attributions_group1"]
        avg_attr_group2 = comparison_results["avg_attributions_group2"]
        attr_diff = comparison_results["attribution_differences"]
        
        # Ensure feature names match the number of features
        if self.feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(avg_attr_group1))]
        else:
            # If provided feature names don't match, truncate or extend
            if len(self.feature_names) != len(avg_attr_group1):
                print(f"Warning: Number of feature names ({len(self.feature_names)}) doesn't match number of features ({len(avg_attr_group1)})")
                if len(self.feature_names) > len(avg_attr_group1):
                    feature_names = self.feature_names[:len(avg_attr_group1)]
                else:
                    feature_names = self.feature_names + [f"Feature {i}" for i in range(len(self.feature_names), len(avg_attr_group1))]
            else:
                feature_names = self.feature_names
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot average attributions for group 1
        ax1.bar(range(len(avg_attr_group1)), avg_attr_group1, color='blue', alpha=0.7)
        ax1.set_xticks(range(len(avg_attr_group1)))
        ax1.set_xticklabels(feature_names, rotation=90)
        ax1.set_title("Average Attributions for Group 1")
        ax1.set_ylabel("Attribution Score")
        
        # Plot average attributions for group 2
        ax2.bar(range(len(avg_attr_group2)), avg_attr_group2, color='green', alpha=0.7)
        ax2.set_xticks(range(len(avg_attr_group2)))
        ax2.set_xticklabels(feature_names, rotation=90)
        ax2.set_title("Average Attributions for Group 2")
        ax2.set_ylabel("Attribution Score")
        
        # Plot attribution differences
        colors = ['blue' if diff >= 0 else 'red' for diff in attr_diff]
        ax3.bar(range(len(attr_diff)), attr_diff, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(attr_diff)))
        ax3.set_xticklabels(feature_names, rotation=90)
        ax3.set_title("Attribution Differences (Group 1 - Group 2)")
        ax3.set_ylabel("Difference in Attribution")
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.95)
        
        return plt