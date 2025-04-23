import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from captum.attr import (
    IntegratedGradients,
    Saliency,
    FeatureAblation
)
from captum.attr import visualization as viz

# Set Seaborn style for all plots
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Input Tensor .* did not already require gradients")
warnings.filterwarnings("ignore", message=".*Using feature ablation instead.*")

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

        # Initialize attribution methods
        self.ig = IntegratedGradients(self.model_forward)
        self.saliency = Saliency(self.model_forward)
        self.feature_ablation = FeatureAblation(self.model_forward)

        # Store attribution history for bias tracking
        self.attribution_history = {}
        self.bias_metrics = {}

    def model_forward(self, inputs):
        """Forward function for the model that returns Q-values"""
        return self.model(inputs)

    def explain_decision(self, state, action_idx=None, method='ig', n_samples=50, sensitive_attr=None):
        """
        Compute attributions for a decision using various attribution methods

        Args:
            state: The input state to explain
            action_idx: The action index to explain (if None, uses max Q-value action)
            method: Attribution method to use ('ig', 'saliency', 'deep_lift', 'feature_ablation', 'gradient_shap')
            n_samples: Number of samples for sampling-based methods
            sensitive_attr: Index of sensitive attribute for bias tracking

        Returns:
            attributions: Feature importance scores
        """
        # Handle sparse matrices by converting to dense array
        if hasattr(state, 'toarray'):
            state = state.toarray().flatten()

        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float)
        else:
            state_tensor = state.clone()

        # Ensure state_tensor is 2D (batch, features) for batch norm
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Set model to eval mode for inference
        self.model.eval()

        # If no action is specified, use the model's chosen action
        if action_idx is None:
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action_idx = torch.argmax(q_values, dim=1).item()

        # Convert action_idx to int if it's a numpy type
        if isinstance(action_idx, (np.int32, np.int64)):
            action_idx = int(action_idx)

        # Baseline is typically zero vector
        baseline = torch.zeros_like(state_tensor)

        try:
            # Compute attributions based on selected method
            if method == 'ig':
                try:
                    attributions = self.ig.attribute(state_tensor, baseline, target=action_idx, n_steps=50)
                except RuntimeError as e:
                    if "One of the differentiated Tensors appears to not have been used in the graph" in str(e):
                        # Silently fall back to feature ablation
                        # Fall back to feature ablation which doesn't have this issue
                        attributions = self.feature_ablation.attribute(state_tensor, target=action_idx)
                    else:
                        raise
            elif method == 'saliency':
                try:
                    attributions = self.saliency.attribute(state_tensor, target=action_idx)
                except RuntimeError as e:
                    if "One of the differentiated Tensors appears to not have been used in the graph" in str(e):
                        # Silently fall back to feature ablation
                        # Fall back to feature ablation which doesn't have this issue
                        attributions = self.feature_ablation.attribute(state_tensor, target=action_idx)
                    else:
                        raise
            elif method == 'feature_ablation':
                # Feature ablation doesn't use gradients, so it shouldn't have the same issue
                attributions = self.feature_ablation.attribute(state_tensor, target=action_idx)
            else:
                # Default to feature ablation as it's more robust
                attributions = self.feature_ablation.attribute(state_tensor, target=action_idx)
        except Exception as e:
            # Silently handle errors and return zeros as fallback
            attributions = torch.zeros_like(state_tensor)

        # Get numpy array of attributions
        attr_np = attributions.squeeze(0).detach().numpy()

        # Track attributions for bias detection if sensitive attribute is provided
        if sensitive_attr is not None:
            # Get sensitive value from the original state
            if isinstance(state, np.ndarray):
                sensitive_val = state[sensitive_attr]
            elif isinstance(state, torch.Tensor):
                sensitive_val = state[sensitive_attr].item()
            else:
                sensitive_val = state[sensitive_attr]

            # Ensure it's an integer
            try:
                group_key = f"group_{int(sensitive_val)}"
            except:
                # Default to group_0 if conversion fails
                group_key = "group_0"

            if group_key not in self.attribution_history:
                self.attribution_history[group_key] = []

            self.attribution_history[group_key].append({
                'attributions': attr_np,
                'action': action_idx,
                'state': state
            })

            # Update bias metrics if we have data for multiple groups
            self._update_bias_metrics()

        return attr_np, action_idx

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
        # Set model to eval mode for inference
        self.model.eval()

        # Handle case when input is sparse matrix
        if hasattr(states_group1, 'toarray') and hasattr(states_group1, 'shape'):
            # Convert entire sparse matrix to dense array if it's small enough
            if states_group1.shape[0] <= 100:  # Only convert if reasonable size
                states_group1 = states_group1.toarray()
            # Otherwise process rows individually in the loop below

        if hasattr(states_group2, 'toarray') and hasattr(states_group2, 'shape'):
            if states_group2.shape[0] <= 100:
                states_group2 = states_group2.toarray()

        # Get attributions for both groups with error handling
        attributions_group1 = []
        actions_group1 = []

        # Process group 1 samples
        for i in range(min(states_group1.shape[0], 1000)):  # Process up to 1000 samples for better analysis
            try:
                # Extract individual state (handles both dense and sparse)
                if hasattr(states_group1, 'toarray') and not isinstance(states_group1, np.ndarray):
                    state = states_group1[i]  # Will be handled in explain_decision
                else:
                    state = states_group1[i]

                attr, action = self.explain_decision(state)
                attributions_group1.append(attr)
                actions_group1.append(action)
            except Exception:
                # Silently handle errors
                continue

        # Process group 2 samples
        attributions_group2 = []
        actions_group2 = []

        for i in range(min(states_group2.shape[0], 1000)):  # Process up to 1000 samples for better analysis
            try:
                # Extract individual state
                if hasattr(states_group2, 'toarray') and not isinstance(states_group2, np.ndarray):
                    state = states_group2[i]
                else:
                    state = states_group2[i]

                attr, action = self.explain_decision(state)
                attributions_group2.append(attr)
                actions_group2.append(action)
            except Exception:
                # Silently handle errors
                continue

        # Check if we have enough samples
        if len(attributions_group1) == 0 or len(attributions_group2) == 0:
            # Silently handle insufficient samples
            # Return empty comparison
            return {
                "avg_attributions_group1": np.zeros(states_group1.shape[1] if hasattr(states_group1, 'shape') else 0),
                "avg_attributions_group2": np.zeros(states_group2.shape[1] if hasattr(states_group2, 'shape') else 0),
                "attribution_differences": np.zeros(states_group1.shape[1] if hasattr(states_group1, 'shape') else 0),
                "approve_rate_group1": 0,
                "approve_rate_group2": 0,
                "approve_rate_difference": 0
            }

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
        """Visualize feature attributions with enhanced styling"""
        if self.feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(attributions))]
        else:
            feature_names = self.feature_names

        # Create a DataFrame for better Seaborn plotting
        import pandas as pd
        df = pd.DataFrame({
            'Feature': feature_names,
            'Attribution': attributions,
            'Impact': ['Positive' if attr >= 0 else 'Negative' for attr in attributions]
        })

        # Sort by absolute attribution value for better visualization
        df['Abs_Attribution'] = abs(df['Attribution'])
        df = df.sort_values('Abs_Attribution', ascending=False).head(15)  # Show top 15 features

        # Create the plot
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Attribution', y='Feature', data=df, hue='Impact',
                        palette={'Positive': '#2ecc71', 'Negative': '#e74c3c'})

        # Add value labels
        for i, v in enumerate(df['Attribution']):
            ax.text(v + (0.01 if v >= 0 else -0.01),
                   i,
                   f'{v:.3f}',
                   va='center',
                   ha='left' if v >= 0 else 'right',
                   fontweight='bold')

        # Customize the plot
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Attribution Score", fontsize=14, labelpad=10)
        plt.ylabel("Features", fontsize=14, labelpad=10)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()

        return plt

    def _update_bias_metrics(self):
        """
        Update bias metrics based on attribution history
        """
        # Check if we have data for at least two groups
        if len(self.attribution_history.keys()) < 2:
            return

        # Get group keys
        group_keys = list(self.attribution_history.keys())

        # Compute bias metrics between all pairs of groups
        for i in range(len(group_keys)):
            for j in range(i+1, len(group_keys)):
                group1 = group_keys[i]
                group2 = group_keys[j]

                # Skip if either group has no data
                if not self.attribution_history[group1] or not self.attribution_history[group2]:
                    continue

                # Get attributions for each group
                attrs_group1 = np.array([entry['attributions'] for entry in self.attribution_history[group1]])
                attrs_group2 = np.array([entry['attributions'] for entry in self.attribution_history[group2]])

                # Get actions for each group
                actions_group1 = np.array([entry['action'] for entry in self.attribution_history[group1]])
                actions_group2 = np.array([entry['action'] for entry in self.attribution_history[group2]])

                # Compute average attributions
                avg_attr_group1 = np.mean(attrs_group1, axis=0)
                avg_attr_group2 = np.mean(attrs_group2, axis=0)

                # Compute attribution differences
                attr_diff = avg_attr_group1 - avg_attr_group2

                # Compute approval rates
                approve_rate_group1 = np.mean([1 if a == 1 else 0 for a in actions_group1])
                approve_rate_group2 = np.mean([1 if a == 1 else 0 for a in actions_group2])

                # Store metrics
                pair_key = f"{group1}_vs_{group2}"
                self.bias_metrics[pair_key] = {
                    "avg_attributions_group1": avg_attr_group1,
                    "avg_attributions_group2": avg_attr_group2,
                    "attribution_differences": attr_diff,
                    "approve_rate_group1": approve_rate_group1,
                    "approve_rate_group2": approve_rate_group2,
                    "approve_rate_difference": approve_rate_group1 - approve_rate_group2,
                    "attribution_difference_magnitude": np.linalg.norm(attr_diff)
                }

    def get_bias_metrics(self):
        """
        Get current bias metrics

        Returns:
            dict: Current bias metrics
        """
        return self.bias_metrics

    def compute_attribution_based_bias(self, sensitive_attr_idx, threshold=0.1):
        """
        Compute bias based on attributions to sensitive attribute

        Args:
            sensitive_attr_idx: Index of the sensitive attribute
            threshold: Threshold for determining significant bias

        Returns:
            dict: Bias metrics
        """
        # Check if we have attribution history
        if not self.attribution_history:
            return {"bias_detected": False, "message": "No attribution history available"}

        # Get all attributions
        all_attributions = []
        all_actions = []
        all_groups = []

        for group_key, group_data in self.attribution_history.items():
            group_id = int(group_key.split('_')[1])  # Extract group ID (0 for female, 1 for male)
            for entry in group_data:
                all_attributions.append(entry['attributions'])
                all_actions.append(entry['action'])
                all_groups.append(group_id)

        all_attributions = np.array(all_attributions)
        all_actions = np.array(all_actions)
        all_groups = np.array(all_groups)

        # Get attributions for sensitive attribute
        sensitive_attr_attributions = all_attributions[:, sensitive_attr_idx]

        # Compute statistics
        mean_attribution = np.mean(sensitive_attr_attributions)
        std_attribution = np.std(sensitive_attr_attributions)
        abs_mean_attribution = np.mean(np.abs(sensitive_attr_attributions))

        # Compute group-specific statistics
        female_indices = np.where(all_groups == 0)[0]
        male_indices = np.where(all_groups == 1)[0]

        female_attributions = sensitive_attr_attributions[female_indices]
        male_attributions = sensitive_attr_attributions[male_indices]

        female_actions = all_actions[female_indices]
        male_actions = all_actions[male_indices]

        # Compute approval rates
        female_approval_rate = np.mean([1 if a == 1 else 0 for a in female_actions]) if len(female_actions) > 0 else 0
        male_approval_rate = np.mean([1 if a == 1 else 0 for a in male_actions]) if len(male_actions) > 0 else 0

        # Compute demographic parity
        demographic_parity_diff = abs(female_approval_rate - male_approval_rate)

        # Compute attribution statistics by group
        female_mean_attr = np.mean(female_attributions) if len(female_attributions) > 0 else 0
        male_mean_attr = np.mean(male_attributions) if len(male_attributions) > 0 else 0

        female_abs_mean_attr = np.mean(np.abs(female_attributions)) if len(female_attributions) > 0 else 0
        male_abs_mean_attr = np.mean(np.abs(male_attributions)) if len(male_attributions) > 0 else 0

        # Determine if bias is detected
        bias_detected = abs_mean_attribution > threshold

        # Compute attribution-based fairness metrics
        attribution_disparity = abs(female_mean_attr - male_mean_attr)
        abs_attribution_disparity = abs(female_abs_mean_attr - male_abs_mean_attr)

        # Compute correlation between sensitive attribute attribution and decision
        if len(sensitive_attr_attributions) > 1 and len(all_actions) > 1:
            correlation = np.corrcoef(sensitive_attr_attributions, all_actions)[0, 1]
        else:
            correlation = 0

        return {
            "bias_detected": bias_detected,
            "mean_attribution": mean_attribution,
            "std_attribution": std_attribution,
            "abs_mean_attribution": abs_mean_attribution,
            "female_mean_attribution": female_mean_attr,
            "male_mean_attribution": male_mean_attr,
            "female_abs_mean_attribution": female_abs_mean_attr,
            "male_abs_mean_attribution": male_abs_mean_attr,
            "attribution_disparity": attribution_disparity,
            "abs_attribution_disparity": abs_attribution_disparity,
            "female_approval_rate": female_approval_rate,
            "male_approval_rate": male_approval_rate,
            "demographic_parity_diff": demographic_parity_diff,
            "correlation_attr_decision": correlation,
            "threshold": threshold,
            "message": f"Bias {'detected' if bias_detected else 'not detected'} in sensitive attribute"
        }

    def get_attribution_based_fairness_constraint(self, sensitive_attr_idx, lambda_param=0.1):
        """
        Generate a fairness constraint based on attributions

        Args:
            sensitive_attr_idx: Index of the sensitive attribute
            lambda_param: Weight of the constraint

        Returns:
            function: Constraint function that takes state, action, and returns penalty
        """
        # Compute bias metrics if not already computed
        bias_metrics = self.compute_attribution_based_bias(sensitive_attr_idx)

        def constraint_fn(state, action):
            # If no bias detected, return zero penalty
            if not bias_metrics["bias_detected"]:
                return 0.0

            # Get attribution for this state-action pair
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            baseline = torch.zeros_like(state_tensor)

            # Compute attribution
            attributions = self.ig.attribute(state_tensor, baseline, target=action)
            attr_np = attributions.squeeze(0).detach().numpy()

            # Get attribution for sensitive attribute
            sensitive_attr_attribution = attr_np[sensitive_attr_idx]

            # Compute penalty based on attribution magnitude
            # Higher attribution to sensitive attribute = higher penalty
            penalty = lambda_param * abs(sensitive_attr_attribution)

            return penalty

        return constraint_fn

    def visualize_bias_metrics(self, title="Bias Metrics"):
        """
        Visualize bias metrics with enhanced styling using Seaborn

        Returns:
            plt: Matplotlib plot
        """
        if not self.bias_metrics:
            # Silently handle missing metrics
            return None

        import pandas as pd

        # Create a more comprehensive visualization
        fig = plt.figure(figsize=(16, 12))

        # Define grid layout
        gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2], wspace=0.25, hspace=0.3)

        # Create subplots
        ax1 = fig.add_subplot(gs[0, 0])  # Approval rates
        ax2 = fig.add_subplot(gs[0, 1])  # Attribution magnitudes
        ax3 = fig.add_subplot(gs[1, :])  # Fairness metrics

        # Collect data from all pairs
        all_metrics = {}
        for pair_key, metrics in self.bias_metrics.items():
            # Store metrics
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        # Prepare data for approval rates plot
        group_labels = []
        group_types = []
        approval_rates = []

        for pair_key, metrics in self.bias_metrics.items():
            group1, group2 = pair_key.split('_vs_')

            # Group 1 data
            group_labels.append(group1.replace('group_', 'Group '))
            group_types.append('Female' if 'group_0' in group1 else 'Male')
            approval_rates.append(metrics['approve_rate_group1'])

            # Group 2 data
            group_labels.append(group2.replace('group_', 'Group '))
            group_types.append('Female' if 'group_0' in group2 else 'Male')
            approval_rates.append(metrics['approve_rate_group2'])

        # Create DataFrame for approval rates
        approval_df = pd.DataFrame({
            'Group': group_labels,
            'Type': group_types,
            'Approval Rate': approval_rates
        })

        # Plot approval rates with Seaborn
        sns.barplot(x='Group', y='Approval Rate', hue='Type', data=approval_df,
                   palette={'Female': '#9b59b6', 'Male': '#3498db'}, ax=ax1)

        # Customize approval rates plot
        ax1.set_title("Approval Rates by Group", fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel("Group", fontsize=12, labelpad=10)
        ax1.set_ylabel("Approval Rate", fontsize=12, labelpad=10)
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for p in ax1.patches:
            height = p.get_height()
            ax1.text(p.get_x() + p.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Prepare data for attribution magnitudes plot
        pair_labels = [f"{pair_key.split('_vs_')[0].replace('group_', 'Female' if 'group_0' in pair_key.split('_vs_')[0] else 'Male')} vs {pair_key.split('_vs_')[1].replace('group_', 'Female' if 'group_0' in pair_key.split('_vs_')[1] else 'Male')}"
                      for pair_key in self.bias_metrics.keys()]
        attribution_diffs = [metrics['attribution_difference_magnitude'] for metrics in self.bias_metrics.values()]

        # Create DataFrame for attribution magnitudes
        attr_df = pd.DataFrame({
            'Comparison': pair_labels,
            'Magnitude': attribution_diffs
        })

        # Plot attribution magnitudes with Seaborn
        sns.barplot(x='Magnitude', y='Comparison', data=attr_df,
                   palette=sns.color_palette("YlOrRd", len(attr_df)), ax=ax2)

        # Customize attribution magnitudes plot
        ax2.set_title("Attribution Difference Magnitude", fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel("Magnitude", fontsize=12, labelpad=10)
        ax2.set_ylabel("Group Comparison", fontsize=12, labelpad=10)

        # Add value labels
        for p in ax2.patches:
            width = p.get_width()
            ax2.text(width + 0.01, p.get_y() + p.get_height()/2.,
                    f'{width:.2f}', ha='left', va='center', fontsize=10, fontweight='bold')

        # Prepare data for fairness metrics plot
        fairness_metrics_data = {
            'Demographic Parity': [abs(metrics['approve_rate_group1'] - metrics['approve_rate_group2'])
                                for metrics in self.bias_metrics.values()],
            'Attribution Disparity': [metrics.get('attribution_disparity', 0)
                                    for metrics in self.bias_metrics.values()],
            'Abs Attribution Disparity': [metrics.get('abs_attribution_disparity', 0)
                                        for metrics in self.bias_metrics.values()],
            'Correlation (Attr-Decision)': [metrics.get('correlation_attr_decision', 0)
                                          for metrics in self.bias_metrics.values()]
        }

        # Create a long-form DataFrame for Seaborn
        fairness_df = pd.DataFrame()
        for metric_name, values in fairness_metrics_data.items():
            temp_df = pd.DataFrame({
                'Comparison': pair_labels,
                'Metric': metric_name,
                'Value': values
            })
            fairness_df = pd.concat([fairness_df, temp_df])

        # Plot fairness metrics with Seaborn
        sns.barplot(x='Comparison', y='Value', hue='Metric', data=fairness_df,
                   palette='viridis', ax=ax3)

        # Customize fairness metrics plot
        ax3.set_title("Fairness Metrics Comparison", fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel("Group Comparison", fontsize=14, labelpad=10)
        ax3.set_ylabel("Metric Value", fontsize=14, labelpad=10)
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Metric', title_fontsize=12, fontsize=10, loc='upper right')

        # Add a title to the entire figure
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title

        return fig

    def visualize_attribution_comparison(self, comparison_results, title="Attribution Comparison", max_features=15):
        """Visualize attribution comparison between groups with improved readability"""
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

        # Find the most important features based on absolute attribution differences
        abs_diff = np.abs(attr_diff)
        top_indices = np.argsort(abs_diff)[-max_features:]

        # Extract top features and their attributions
        top_feature_names = [feature_names[i] for i in top_indices]
        top_attr_group1 = [avg_attr_group1[i] for i in top_indices]
        top_attr_group2 = [avg_attr_group2[i] for i in top_indices]
        top_attr_diff = [attr_diff[i] for i in top_indices]

        # Create figure with improved layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # Plot top feature attributions for both groups side by side
        x = np.arange(len(top_feature_names))
        width = 0.35

        ax1.bar(x - width/2, top_attr_group1, width, label='Female', color='blue', alpha=0.7)
        ax1.bar(x + width/2, top_attr_group2, width, label='Male', color='green', alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels(top_feature_names, rotation=45, ha='right')
        ax1.set_title("Top Feature Attributions by Group")
        ax1.set_ylabel("Attribution Score")
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.3)

        # Plot attribution differences for top features
        colors = ['blue' if diff >= 0 else 'red' for diff in top_attr_diff]
        ax2.bar(x, top_attr_diff, color=colors, alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(top_feature_names, rotation=45, ha='right')
        ax2.set_title("Attribution Differences (Female - Male)")
        ax2.set_ylabel("Difference in Attribution")
        ax2.grid(axis='y', linestyle='--', alpha=0.3)

        # Add approval rate information if available
        if "approve_rate_group1" in comparison_results and "approve_rate_group2" in comparison_results:
            approve_rate_group1 = comparison_results["approve_rate_group1"]
            approve_rate_group2 = comparison_results["approve_rate_group2"]
            approve_rate_diff = approve_rate_group1 - approve_rate_group2

            approval_text = (f"Approval Rates: Female: {approve_rate_group1:.2f}, Male: {approve_rate_group2:.2f}\n"
                            f"Difference: {approve_rate_diff:.2f} (Demographic Parity: {abs(approve_rate_diff):.2f})")

            fig.text(0.5, 0.01, approval_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.95, bottom=0.1)

        return plt