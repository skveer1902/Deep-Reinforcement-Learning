import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import random
import argparse
import warnings
from tqdm import tqdm

# Create directories for results if they don't exist
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# Set Seaborn style for all plots
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Input Tensor .* did not already require gradients")
warnings.filterwarnings("ignore", message=".*Using feature ablation instead.*")

# Import project modules
from fairrl.data.loader import AdultDataset
from fairrl.models.environment import LoanEnvironment
from fairrl.models.models import DDQNAgent
from fairrl.explainability.ig import IGExplainer
from fairrl.fairness.metrics import FairnessMetrics
from fairrl.fairness.constraints import (
    DemographicParityConstraint,
    EqualizedOddsConstraint,
    EqualOpportunityConstraint,
    AttributionBasedConstraint,
    ConstraintOptimizer
)

def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_baseline_model(env, agent, num_episodes=1000, seed=42):
    """Train a baseline RL model without fairness constraints"""
    set_seeds(seed)

    scores = []
    scores_window = deque(maxlen=100)
    decisions = []

    for i_episode in tqdm(range(1, num_episodes+1), desc="Training Baseline Model"):
        state = env.reset()
        score = 0

        # Interact with environment (single step for our loan environment)
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        # For consistency with the fair model, pass None as fairness_info
        agent.step(state, action, reward, next_state, done, None)

        score += reward

        # Store decision information
        decisions.append({
            'episode': i_episode,
            'action': action,
            'reward': reward,
            'sensitive': info['sensitive'],
            'ground_truth': info['ground_truth'],
            'correct_decision': info['correct_decision']
        })

        # Update progress tracking
        scores_window.append(score)
        scores.append(score)

        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode} Average Score: {np.mean(scores_window):.2f}')

            # Get fairness stats from agent if available
            if hasattr(agent, 'get_fairness_stats'):
                stats = agent.get_fairness_stats()
                print(f"Buffer size: {stats['buffer_size']}")

    return agent, np.array(scores), pd.DataFrame(decisions)

def train_fair_model(env, agent, constraint_optimizer, num_episodes=1000, seed=42, curriculum_learning=True):
    """Train a fair RL model with fairness constraints and curriculum learning"""
    set_seeds(seed)

    scores = []
    scores_window = deque(maxlen=100)
    decisions = []

    # For curriculum learning - balanced approach for both metrics
    # Different weight schedules for different constraints
    dp_weight_schedule = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0]  # High weights for demographic parity
    eo_weight_schedule = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]  # Increased weights for equal opportunity
    attr_weight_schedule = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Lower weights for attribution
    curriculum_phase = 0
    phase_length = num_episodes // len(dp_weight_schedule)

    # Track fairness violations for analysis
    fairness_violations = 0
    total_decisions = 0

    for i_episode in tqdm(range(1, num_episodes+1), desc="Training Fair Model"):
        state = env.reset()
        score = 0

        # Update constraint weights for curriculum learning with different schedules
        if curriculum_learning and i_episode % phase_length == 0 and curriculum_phase < len(dp_weight_schedule) - 1:
            curriculum_phase += 1
            dp_weight = dp_weight_schedule[curriculum_phase]
            eo_weight = eo_weight_schedule[curriculum_phase]
            attr_weight = attr_weight_schedule[curriculum_phase]
            print(f"\nCurriculum phase {curriculum_phase}: DP weight={dp_weight}, EO weight={eo_weight}, Attr weight={attr_weight}")

            # Set different weights for different constraint types
            for constraint in constraint_optimizer.constraints:
                if isinstance(constraint, DemographicParityConstraint):
                    constraint.lambda_param = dp_weight
                elif isinstance(constraint, EqualOpportunityConstraint):
                    constraint.lambda_param = eo_weight
                elif isinstance(constraint, AttributionBasedConstraint):
                    constraint.lambda_param = attr_weight
                else:
                    # Default weight for other constraints
                    constraint.lambda_param = eo_weight

        # Interact with environment (single step for our loan environment)
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        # Apply fairness constraints to modify reward
        sensitive = info['sensitive']
        total_decisions += 1

        # Get ground_truth if available, otherwise set to None
        ground_truth = info.get('ground_truth', None)

        # Apply fairness constraints with safety checks
        try:
            fair_reward = constraint_optimizer.compute_fair_reward(
                state, action, reward, sensitive, ground_truth
            )

            # Check if this is a fairness violation - balanced approach for both metrics
            fairness_violation = False
            penalty = reward - fair_reward  # Positive penalty means the fair_reward is lower than original reward

            # Get sensitive attribute and action for more precise detection
            is_female = sensitive == 0
            is_approve = action == 1
            is_qualified = ground_truth == 1  # For equal opportunity

            # Get constraint-specific information
            dp_info = None
            eo_info = None
            for constraint in constraint_optimizer.constraints:
                if isinstance(constraint, DemographicParityConstraint):
                    dp_info = constraint.get_metrics()
                elif isinstance(constraint, EqualOpportunityConstraint):
                    eo_info = constraint.get_metrics()

            # Track violation type for better prioritization
            violation_type = None

            # Demographic parity violation detection
            if dp_info and 'disparity' in dp_info:
                # Detect demographic parity violations specifically
                if is_female and not is_approve and dp_info.get('female_rate', 0) < dp_info.get('male_rate', 0):
                    # Denying a female when female approval rate is already lower
                    fairness_violation = True
                    violation_type = 'demographic_parity'
                    fairness_violations += 1
                elif not is_female and is_approve and dp_info.get('male_rate', 0) > dp_info.get('female_rate', 0):
                    # Approving a male when male approval rate is already higher
                    fairness_violation = True
                    violation_type = 'demographic_parity'
                    fairness_violations += 1

            # Equal opportunity violation detection (for qualified applicants)
            if not fairness_violation and is_qualified and eo_info and 'tpr_disparity' in eo_info:
                # Detect equal opportunity violations specifically
                if is_female and not is_approve and eo_info.get('tpr_female', 0) < eo_info.get('tpr_male', 0):
                    # Denying a qualified female when female TPR is already lower
                    fairness_violation = True
                    violation_type = 'equal_opportunity'
                    fairness_violations += 1
                elif not is_female and not is_approve and eo_info.get('tpr_male', 0) < eo_info.get('tpr_female', 0):
                    # Denying a qualified male when male TPR is already lower
                    fairness_violation = True
                    violation_type = 'equal_opportunity'
                    fairness_violations += 1

            # Also consider general penalties as a fallback
            if not fairness_violation and penalty > 0.8:  # Significant penalty indicates fairness violation
                fairness_violation = True
                violation_type = 'general'
                fairness_violations += 1

            # Ensure reward is within reasonable bounds to prevent numerical instability
            if not np.isfinite(fair_reward):
                print(f"Warning: Non-finite reward detected: {fair_reward}. Using original reward instead.")
                fair_reward = reward
                fairness_violation = False
            elif abs(fair_reward) > 100:
                print(f"Warning: Extreme reward detected: {fair_reward}. Clipping to range [-100, 100].")
                fair_reward = np.clip(fair_reward, -100, 100)

            # Create fairness info for the replay buffer
            fairness_info = {
                'fairness_violation': fairness_violation,
                'violation_type': violation_type,  # Include violation type for better prioritization
                'penalty': penalty,
                'sensitive': sensitive,
                'ground_truth': ground_truth
            }
        except Exception as e:
            print(f"Error computing fair reward: {e}. Using original reward instead.")
            fair_reward = reward
            fairness_info = None

        # Update agent with fair reward and fairness info
        agent.step(state, action, fair_reward, next_state, done, fairness_info)

        score += fair_reward

        # Store decision information
        decisions.append({
            'episode': i_episode,
            'action': action,
            'reward': reward,
            'fair_reward': fair_reward,
            'sensitive': sensitive,
            'ground_truth': ground_truth,
            'correct_decision': info['correct_decision'],
            'fairness_violation': fairness_info['fairness_violation'] if fairness_info else False
        })

        # Update progress tracking
        scores_window.append(score)
        scores.append(score)

        # Update constraint weights
        if not curriculum_learning:  # Only use the built-in weight update if not using curriculum learning
            constraint_optimizer.update_constraint_weights(i_episode)

        # Print progress and fairness statistics
        if i_episode % 100 == 0:
            violation_rate = fairness_violations / max(1, total_decisions)
            print(f'\rEpisode {i_episode} | Avg Score: {np.mean(scores_window):.2f} | Fairness Violations: {violation_rate:.2%}')

            # Get fairness stats from agent if available
            if hasattr(agent, 'get_fairness_stats'):
                stats = agent.get_fairness_stats()
                if 'fairness_sample_ratio' in stats:
                    print(f"Fairness sample ratio: {stats['fairness_sample_ratio']:.2f} | Buffer violations: {stats['buffer_violation_count']}")

    # Final fairness statistics
    violation_rate = fairness_violations / max(1, total_decisions)
    print(f"\nTraining complete. Final fairness violation rate: {violation_rate:.2%}")

    return agent, np.array(scores), pd.DataFrame(decisions)

def evaluate_model(env, agent, num_episodes=2000, seed=42):
    """Evaluate a trained RL model"""
    set_seeds(seed)

    scores = []
    decisions = []

    for i_episode in tqdm(range(1, num_episodes+1), desc="Evaluating Model"):
        state = env.reset()

        # Interact with environment (no exploration)
        action = agent.act(state, training=False)
        _, reward, _, info = env.step(action)

        scores.append(reward)

        # Store decision information
        decisions.append({
            'action': action,
            'reward': reward,
            'sensitive': info['sensitive'],
            'ground_truth': info['ground_truth'],
            'correct_decision': info['correct_decision']
        })

    # Convert to DataFrame for analysis
    decisions_df = pd.DataFrame(decisions)

    # Calculate overall metrics
    accuracy = decisions_df['correct_decision'].mean()
    avg_reward = decisions_df['reward'].mean()

    # Calculate metrics by sensitive attribute
    group_metrics = decisions_df.groupby('sensitive').agg({
        'action': 'mean',  # Approval rate
        'correct_decision': 'mean',  # Accuracy
        'reward': 'mean'  # Average reward
    }).reset_index()

    # Rename for clarity
    group_metrics = group_metrics.rename(columns={
        'action': 'approval_rate',
        'correct_decision': 'accuracy',
        'reward': 'avg_reward'
    })

    # Add group labels
    group_metrics['group'] = group_metrics['sensitive'].apply(
        lambda x: 'Female' if x == 0 else 'Male'
    )

    return {
        'decisions': decisions_df,
        'overall_accuracy': accuracy,
        'overall_reward': avg_reward,
        'group_metrics': group_metrics
    }

def analyze_fairness(evaluation_results, fairness_metrics):
    """Analyze fairness metrics from evaluation results"""
    decisions = evaluation_results['decisions']

    # Extract data for fairness analysis
    y_true = decisions['ground_truth'].values
    y_pred = decisions['action'].values
    sensitive = decisions['sensitive'].values

    # Calculate fairness metrics
    demographic_parity = fairness_metrics.demographic_parity(y_pred, sensitive)
    equal_odds = fairness_metrics.equalized_odds(y_true, y_pred, sensitive)
    equal_opportunity = fairness_metrics.equal_opportunity(y_true, y_pred, sensitive)

    # Combine all metrics
    all_metrics = {
        **demographic_parity,
        **equal_odds,
        **equal_opportunity
    }

    return all_metrics

def analyze_bias_with_ig(model, feature_names, dataset, fairness_metrics, num_samples=1000):
    """Analyze bias using Integrated Gradients and other attribution methods"""
    # Initialize explainer
    explainer = IGExplainer(model, feature_names[:-1])  # Remove sensitive attribute from feature names

    # Get samples for analysis - using a smaller number of samples for stability
    try:
        X, y, sensitive, _ = dataset.preprocess_data()

        # Split by sensitive attribute
        female_indices = np.where(sensitive == 0)[0][:num_samples]
        male_indices = np.where(sensitive == 1)[0][:num_samples]

        # Ensure we have enough samples
        if len(female_indices) == 0 or len(male_indices) == 0:
            print("Warning: Not enough samples for one or both groups")
            # Create dummy samples if needed
            if len(female_indices) == 0:
                print("Creating dummy female samples")
                female_indices = np.where(sensitive == 1)[0][:min(5, len(np.where(sensitive == 1)[0]))]
            if len(male_indices) == 0:
                print("Creating dummy male samples")
                male_indices = np.where(sensitive == 0)[0][:min(5, len(np.where(sensitive == 0)[0]))]

        # Get samples
        female_samples_full = X[female_indices]
        male_samples_full = X[male_indices]

        # Remove sensitive attribute from samples (last column)
        sensitive_attr_idx = X.shape[1] - 1
        female_samples = np.delete(female_samples_full, sensitive_attr_idx, axis=1)
        male_samples = np.delete(male_samples_full, sensitive_attr_idx, axis=1)

        print(f"Analyzing {len(female_samples)} female samples and {len(male_samples)} male samples")
    except Exception as e:
        print(f"Error preparing samples for analysis: {e}")
        # Create dummy samples
        feature_count = len(feature_names) - 1  # Adjust for removed sensitive attribute
        female_samples = np.zeros((5, feature_count))
        male_samples = np.zeros((5, feature_count))

    # Since we've removed the sensitive attribute from the feature names and samples,
    # we don't need to find its index for attribution analysis
    # We'll use a dummy value for the bias metrics computation
    sensitive_attr_idx = 0  # This won't be used for feature attribution since the feature is removed

    print("Sensitive attribute (sex) has been removed from the model inputs for fairness")

    # Compare attributions between groups using different methods
    results = {}
    plots = {}

    # Use different attribution methods with error handling
    for method in ['ig', 'saliency', 'feature_ablation']:
        try:
            # Compare attributions between groups with specific method
            # First, get attributions for each sample using the current method
            female_attributions = []
            male_attributions = []

            # Fix: Use shape[0] instead of len() for sparse matrices
            female_sample_count = female_samples.shape[0]
            male_sample_count = male_samples.shape[0]

            # Process female samples with error handling
            for i in range(min(20, female_sample_count)):  # Use more samples for better analysis
                try:
                    attr, _ = explainer.explain_decision(female_samples[i], method=method)
                    female_attributions.append(attr)
                except Exception as e:
                    print(f"Error processing female sample {i} with method {method}: {e}")
                    # Add zeros as fallback
                    female_attributions.append(np.zeros(len(feature_names)))

            # Process male samples with error handling
            for i in range(min(20, male_sample_count)):
                try:
                    attr, _ = explainer.explain_decision(male_samples[i], method=method)
                    male_attributions.append(attr)
                except Exception as e:
                    print(f"Error processing male sample {i} with method {method}: {e}")
                    # Add zeros as fallback
                    male_attributions.append(np.zeros(len(feature_names)))

            # Check if we have enough samples
            if len(female_attributions) == 0 or len(male_attributions) == 0:
                print(f"Warning: Not enough valid samples for method {method}")
                continue

            # Convert to arrays
            female_attributions = np.array(female_attributions)
            male_attributions = np.array(male_attributions)

            # Compute average attributions
            avg_attr_female = np.mean(female_attributions, axis=0)
            avg_attr_male = np.mean(male_attributions, axis=0)

            # Create comparison dict similar to compare_attributions output
            comparison = {
                "avg_attributions_group1": avg_attr_female,
                "avg_attributions_group2": avg_attr_male,
                "attribution_differences": avg_attr_female - avg_attr_male,
                "method": method
            }

            # Visualize attribution comparison
            plt_comparison = explainer.visualize_attribution_comparison(
                comparison, title=f"Female vs Male Attribution Comparison ({method})"
            )

            results[method] = comparison
            plots[f"{method}_comparison"] = plt_comparison

            # Save method-specific plots
            plt_comparison.savefig(f'results/plots/attribution_comparison_{method}.png')
        except Exception as e:
            print(f"Error analyzing bias with method {method}: {e}")

    # Compute bias metrics based on attributions with error handling
    try:
        bias_metrics = explainer.compute_attribution_based_bias(
            sensitive_attr_idx, threshold=0.1
        )

        # Visualize bias metrics
        try:
            bias_metrics_plot = explainer.visualize_bias_metrics()
            if bias_metrics_plot:
                plots['bias_metrics'] = bias_metrics_plot
                bias_metrics_plot.savefig('results/plots/bias_metrics.png')
        except Exception as e:
            print(f"Error visualizing bias metrics: {e}")
    except Exception as e:
        print(f"Error computing bias metrics: {e}")
        # Create dummy bias metrics
        bias_metrics = {
            "bias_detected": False,
            "mean_attribution": 0,
            "std_attribution": 0,
            "abs_mean_attribution": 0,
            "threshold": 0.1,
            "message": "Error computing bias metrics"
        }

    return {
        'attribution_comparison': results,
        'bias_metrics': bias_metrics,
        'plots': plots
    }

def main(args=None):
    """Main function to run the FairRL pipeline

    Args:
        args: Command-line arguments
    """
    # Create directories for results
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)

    # 1. Load and preprocess the UCI Adult dataset
    print("Loading and preprocessing UCI Adult dataset...")
    dataset = AdultDataset()
    X, y, sensitive, preprocessor = dataset.preprocess_data()
    feature_names = dataset.get_feature_names(preprocessor)

    # Ensure feature names match the number of features
    if len(feature_names) != X.shape[1]:
        print(f"Warning: Number of feature names ({len(feature_names)}) doesn't match number of features ({X.shape[1]})")
        if len(feature_names) > X.shape[1]:
            feature_names = feature_names[:X.shape[1]]
        else:
            feature_names = feature_names + [f"Feature {i}" for i in range(len(feature_names), X.shape[1])]

    print(f"Number of features: {X.shape[1]}")
    print(f"Number of feature names: {len(feature_names)}")

    # 2. Set up RL environment
    print("Setting up RL environment...")
    # Explicitly remove sensitive attribute from state for both baseline and fair models
    env = LoanEnvironment(X, y, sensitive, remove_sensitive=True)

    # Set up model dimensions
    full_state_size = X.shape[1]
    state_size = full_state_size - 1  # Adjust for removed sensitive attribute
    action_size = 2  # Approve or deny

    # Initialize agents with fairness-aware replay buffer for fair model only
    baseline_agent = DDQNAgent(state_size, action_size, fairness_aware=False)

    # Check if we should load pre-trained models or train new ones
    if args and args.load_models:
        print(f"Loading pre-trained baseline model from {args.baseline_model}...")
        try:
            baseline_agent.q_network.load_state_dict(torch.load(args.baseline_model))
            baseline_agent.target_network.load_state_dict(torch.load(args.baseline_model))
            print("Baseline model loaded successfully.")
        except Exception as e:
            print(f"Error loading baseline model: {e}")
            print("Training a new baseline model instead...")
            baseline_agent, _, _ = train_baseline_model(
                env, baseline_agent, num_episodes=8000  # Increased episodes for better learning
            )
            # Save baseline model
            torch.save(baseline_agent.q_network.state_dict(), 'results/models/baseline_model.pth')
    else:
        # 3. Train baseline RL model
        print("Training baseline RL model...")
        baseline_agent, _, _ = train_baseline_model(
            env, baseline_agent, num_episodes=8000  # Increased episodes for better learning
        )
        # Save baseline model
        torch.save(baseline_agent.q_network.state_dict(), 'results/models/baseline_model.pth')

    # 4. Evaluate baseline model
    print("Evaluating baseline model...")
    baseline_evaluation = evaluate_model(env, baseline_agent)

    # 5. Analyze fairness of baseline model
    print("Analyzing fairness of baseline model...")
    fairness_metrics = FairnessMetrics()
    baseline_fairness = analyze_fairness(baseline_evaluation, fairness_metrics)

    # Visualize baseline fairness metrics
    baseline_fairness_plot = fairness_metrics.visualize_fairness_metrics(
        baseline_fairness, title="Baseline Model Fairness Metrics"
    )
    baseline_fairness_plot.savefig('results/plots/baseline_fairness.png')

    # 6. Analyze bias with Integrated Gradients
    print("Analyzing bias with Integrated Gradients...")
    baseline_bias_analysis = analyze_bias_with_ig(
        baseline_agent.q_network, feature_names, dataset, fairness_metrics
    )
    # Save the main IG plot for backward compatibility
    if 'ig_comparison' in baseline_bias_analysis['plots']:
        baseline_bias_analysis['plots']['ig_comparison'].savefig(
            'results/plots/baseline_attribution_comparison.png'
        )

    # Save bias metrics if available
    if 'bias_metrics' in baseline_bias_analysis:
        bias_metrics_df = pd.DataFrame([baseline_bias_analysis['bias_metrics']])
        bias_metrics_df.to_csv('results/baseline_bias_metrics.csv', index=False)

    # Create fair environment with the same configuration as baseline
    # Both models should not have access to the sensitive attribute
    fair_env = LoanEnvironment(X, y, sensitive, remove_sensitive=True)

    # Initialize fair agent with fairness-aware replay buffer
    fair_agent = DDQNAgent(state_size, action_size, fairness_aware=True)  # State size already adjusted for removed sensitive attribute

    # Set up fairness constraints - balanced approach for both metrics
    # Use high weights for both demographic parity and equal opportunity
    demographic_parity_constraint = DemographicParityConstraint(lambda_param=1.0)  # High weight for demographic parity
    equal_opportunity_constraint = EqualOpportunityConstraint(lambda_param=0.5)  # Increased weight for equal opportunity

    # Find the sensitive attribute index
    sensitive_attr_idx = None

    # First, try to find exact match for 'sex'
    for i, name in enumerate(feature_names):
        if name == 'sex':
            sensitive_attr_idx = i
            print(f"Found 'sex' at index {sensitive_attr_idx}")
            break

    # If not found, check if it's the last feature
    if sensitive_attr_idx is None:
        last_idx = len(feature_names) - 1
        if feature_names[last_idx] == 'sex':
            sensitive_attr_idx = last_idx
            print(f"Found 'sex' as the last feature at index {sensitive_attr_idx}")

    # If still not found, look for partial matches
    if sensitive_attr_idx is None:
        for i, name in enumerate(feature_names):
            if 'sex' in name.lower():
                sensitive_attr_idx = i
                print(f"Found feature containing 'sex' at index {sensitive_attr_idx}: {name}")
                break

    # If still not found, use a default value based on the dataset structure
    if sensitive_attr_idx is None:
        # For Adult dataset, sex is typically one of the first few features
        # Let's check the first 15 features and print them for inspection
        print("Could not find 'sex' in feature names. Examining first 15 features:")
        for i, name in enumerate(feature_names[:15]):
            print(f"  {i}: {name}")

        # Default to a common position for sex in Adult dataset
        sensitive_attr_idx = 9
        print(f"Using default sensitive attribute index: {sensitive_attr_idx}")

    print(f"Using sensitive attribute index: {sensitive_attr_idx}, feature name: {feature_names[sensitive_attr_idx]}")

    # Create attribution-based constraint with lower weight to focus on demographic parity
    attribution_constraint = AttributionBasedConstraint(
        fair_agent.q_network,
        sensitive_attr_idx=sensitive_attr_idx,
        lambda_param=0.05,  # Lower weight to focus on demographic parity
        attribution_threshold=0.1  # Higher threshold for less aggressive bias detection
    )

    # Create constraint optimizer with all constraints
    constraint_optimizer = ConstraintOptimizer(
        fair_agent, [
            demographic_parity_constraint,
            equal_opportunity_constraint,  # Added equal opportunity constraint
            attribution_constraint
        ]
    )

    # Check if we should load pre-trained fair model or train a new one
    if args and args.load_models:
        print(f"Loading pre-trained fair model from {args.fair_model}...")
        try:
            fair_agent.q_network.load_state_dict(torch.load(args.fair_model))
            fair_agent.target_network.load_state_dict(torch.load(args.fair_model))
            print("Fair model loaded successfully.")
        except Exception as e:
            print(f"Error loading fair model: {e}")
            print("Training a new fair model instead...")
            # Train with fairness constraints and curriculum learning
            fair_agent, _, _ = train_fair_model(
                fair_env, fair_agent, constraint_optimizer,
                num_episodes=20000,  # Even more episodes for better learning
                curriculum_learning=True  # Enable curriculum learning
            )
            # Save fair model
            torch.save(fair_agent.q_network.state_dict(), 'results/models/fair_model.pth')
    else:
        # Train with fairness constraints for more episodes
        print("Training fair RL model with constraints and curriculum learning...")
        fair_agent, _, _ = train_fair_model(
            fair_env, fair_agent, constraint_optimizer,
            num_episodes=20000,  # Even more episodes for better learning
            curriculum_learning=True  # Enable curriculum learning
        )
        # Save fair model
        torch.save(fair_agent.q_network.state_dict(), 'results/models/fair_model.pth')

    # 8. Evaluate fair model
    print("Evaluating fair model...")
    fair_evaluation = evaluate_model(fair_env, fair_agent)

    # 9. Analyze fairness of fair model
    print("Analyzing fairness of fair model...")
    fair_fairness = analyze_fairness(fair_evaluation, fairness_metrics)

    # Visualize fair model fairness metrics
    fair_fairness_plot = fairness_metrics.visualize_fairness_metrics(
        fair_fairness, title="Fair Model Fairness Metrics"
    )
    fair_fairness_plot.savefig('results/plots/fair_fairness.png')

    # 10. Analyze bias with Integrated Gradients for fair model
    print("Analyzing bias with Integrated Gradients for fair model...")
    fair_bias_analysis = analyze_bias_with_ig(
        fair_agent.q_network, feature_names, dataset, fairness_metrics
    )
    # Save the main IG plot for backward compatibility
    if 'ig_comparison' in fair_bias_analysis['plots']:
        fair_bias_analysis['plots']['ig_comparison'].savefig(
            'results/plots/fair_attribution_comparison.png'
        )

    # Save bias metrics if available
    if 'bias_metrics' in fair_bias_analysis:
        bias_metrics_df = pd.DataFrame([fair_bias_analysis['bias_metrics']])
        bias_metrics_df.to_csv('results/fair_bias_metrics.csv', index=False)

    # Compare bias metrics between baseline and fair models
    if 'bias_metrics' in baseline_bias_analysis and 'bias_metrics' in fair_bias_analysis:
        bias_comparison = pd.DataFrame({
            'Metric': list(baseline_bias_analysis['bias_metrics'].keys()),
            'Baseline': list(baseline_bias_analysis['bias_metrics'].values()),
            'Fair': list(fair_bias_analysis['bias_metrics'].values())
        })
        bias_comparison.to_csv('results/bias_metrics_comparison.csv', index=False)

    # 11. Get constraint statistics
    print("Getting constraint statistics...")
    constraint_stats = constraint_optimizer.get_constraint_stats()

    # Save constraint statistics
    constraint_stats_df = pd.DataFrame()
    for constraint_name, stats in constraint_stats.items():
        if isinstance(stats, dict):
            # Convert nested dict to DataFrame
            stats_df = pd.DataFrame([stats])
            stats_df['constraint'] = constraint_name
            constraint_stats_df = pd.concat([constraint_stats_df, stats_df], ignore_index=True)

    constraint_stats_df.to_csv('results/constraint_stats.csv', index=False)

    # 12. Compare baseline and fair models
    print("Comparing baseline and fair models...")
    # Compare fairness metrics
    fairness_comparison = pd.DataFrame({
        'Metric': list(baseline_fairness.keys()),
        'Baseline': list(baseline_fairness.values()),
        'Fair': list(fair_fairness.values())
    })

    # Create a specific comparison for demographic parity and equal opportunity
    key_metrics = ['demographic_parity_diff', 'equal_opportunity_diff']
    key_metrics_names = ['Demographic Parity', 'Equal Opportunity']

    # Create a comprehensive comparison including both fairness and utility metrics
    key_fairness_comparison = pd.DataFrame({
        'Metric': key_metrics_names,
        'Baseline': [abs(baseline_fairness.get(k, float('nan'))) for k in key_metrics],
        'Fair': [abs(fair_fairness.get(k, float('nan'))) for k in key_metrics],
        'Improvement': [abs(baseline_fairness.get(k, 0)) - abs(fair_fairness.get(k, 0)) for k in key_metrics]
    })

    # Create utility comparison
    utility_metrics = ['Accuracy', 'Reward']
    utility_values = {
        'Baseline': [baseline_evaluation['overall_accuracy'], baseline_evaluation['overall_reward']],
        'Fair': [fair_evaluation['overall_accuracy'], fair_evaluation['overall_reward']],
        'Change': [fair_evaluation['overall_accuracy'] - baseline_evaluation['overall_accuracy'],
                  fair_evaluation['overall_reward'] - baseline_evaluation['overall_reward']]
    }

    utility_comparison = pd.DataFrame({
        'Metric': utility_metrics,
        'Baseline': utility_values['Baseline'],
        'Fair': utility_values['Fair'],
        'Change': utility_values['Change']
    })

    # Print both comparisons
    print("\nKey Fairness Metrics Comparison (lower is better):")
    print(key_fairness_comparison)

    print("\nUtility Metrics Comparison (higher is better):")
    print(utility_comparison)

    # Print a combined summary
    print("\nFairness-Utility Trade-off Summary:")
    fairness_improvement = sum(key_fairness_comparison['Improvement'])
    utility_change = utility_comparison['Change'][0]  # Using accuracy as the main utility metric
    print(f"Overall fairness improvement: {fairness_improvement:.4f}")
    print(f"Accuracy change: {utility_change:.4f}")
    print(f"Trade-off ratio (fairness gain / accuracy change): {fairness_improvement/abs(utility_change) if utility_change != 0 else 'N/A'}")

    # Create enhanced visualizations using Seaborn
    # 1. Fairness metrics comparison
    plt.figure(figsize=(14, 10))

    # Create a melted dataframe for easier plotting with Seaborn
    fairness_melted = pd.melt(key_fairness_comparison,
                             id_vars=['Metric'],
                             value_vars=['Baseline', 'Fair'],
                             var_name='Model', value_name='Value')

    # Create the fairness comparison plot
    plt.subplot(2, 2, 1)
    ax1 = sns.barplot(x='Metric', y='Value', hue='Model', data=fairness_melted,
                    palette={'Baseline': '#3498db', 'Fair': '#2ecc71'})

    # Add value labels
    for i, p in enumerate(ax1.patches):
        height = p.get_height()
        ax1.text(p.get_x() + p.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.title('Fairness Metrics Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Absolute Difference', fontsize=12)
    plt.xlabel('')
    plt.legend(title='Model')

    # 2. Fairness improvement plot
    plt.subplot(2, 2, 2)
    improvement_df = pd.DataFrame({
        'Metric': key_fairness_comparison['Metric'],
        'Improvement': key_fairness_comparison['Improvement']
    })

    # Use colors based on whether there was improvement or not
    colors = ['#2ecc71' if val > 0 else '#e74c3c' for val in improvement_df['Improvement']]

    ax2 = sns.barplot(x='Metric', y='Improvement', data=improvement_df, palette=colors)

    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add value labels
    for i, p in enumerate(ax2.patches):
        height = p.get_height()
        ax2.text(p.get_x() + p.get_width()/2.,
                height + 0.01 if height >= 0 else height - 0.03,
                f'{height:.3f}', ha='center',
                va='bottom' if height >= 0 else 'top',
                fontweight='bold')

    plt.title('Fairness Improvement', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Improvement (Baseline - Fair)', fontsize=12)
    plt.xlabel('')

    # 3. Utility metrics comparison
    plt.subplot(2, 2, 3)
    utility_melted = pd.melt(utility_comparison,
                           id_vars=['Metric'],
                           value_vars=['Baseline', 'Fair'],
                           var_name='Model', value_name='Value')

    ax3 = sns.barplot(x='Metric', y='Value', hue='Model', data=utility_melted,
                    palette={'Baseline': '#3498db', 'Fair': '#2ecc71'})

    # Add value labels
    for i, p in enumerate(ax3.patches):
        height = p.get_height()
        ax3.text(p.get_x() + p.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.title('Utility Metrics Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Value', fontsize=12)
    plt.xlabel('')
    plt.legend(title='Model')

    # 4. Trade-off visualization
    plt.subplot(2, 2, 4)
    trade_off_df = pd.DataFrame({
        'Metric': ['Fairness Improvement', 'Accuracy Change'],
        'Value': [fairness_improvement, utility_change],
        'Type': ['Fairness', 'Utility']
    })

    # Use colors based on whether the change is positive or negative
    colors = ['#2ecc71' if val > 0 else '#e74c3c' for val in trade_off_df['Value']]

    ax4 = sns.barplot(x='Metric', y='Value', data=trade_off_df, palette=colors)

    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add value labels
    for i, p in enumerate(ax4.patches):
        height = p.get_height()
        ax4.text(p.get_x() + p.get_width()/2.,
                height + 0.01 if height >= 0 else height - 0.03,
                f'{height:.3f}', ha='center',
                va='bottom' if height >= 0 else 'top',
                fontweight='bold')

    plt.title('Fairness-Utility Trade-off', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Change', fontsize=12)
    plt.xlabel('')

    # Add a title to the entire figure
    plt.suptitle('Fairness and Utility Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title

    # Save the figure
    plt.savefig('results/figures/fairness_utility_comparison.png', dpi=300, bbox_inches='tight')

    # Create a radar chart for comprehensive comparison
    plt.figure(figsize=(10, 8))

    # Prepare data for radar chart
    categories = ['Demographic Parity', 'Equal Opportunity', 'Accuracy', 'Reward']

    # Normalize values to 0-1 scale for better visualization
    baseline_values = [
        1 - abs(baseline_fairness.get('demographic_parity_diff', 0)),  # Invert so higher is better
        1 - abs(baseline_fairness.get('equal_opportunity_diff', 0)),    # Invert so higher is better
        baseline_evaluation['overall_accuracy'],
        baseline_evaluation['overall_reward'] / 1.0  # Normalize reward
    ]

    fair_values = [
        1 - abs(fair_fairness.get('demographic_parity_diff', 0)),  # Invert so higher is better
        1 - abs(fair_fairness.get('equal_opportunity_diff', 0)),    # Invert so higher is better
        fair_evaluation['overall_accuracy'],
        fair_evaluation['overall_reward'] / 1.0  # Normalize reward
    ]

    # Set up the radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    baseline_values += baseline_values[:1]  # Close the loop
    fair_values += fair_values[:1]  # Close the loop

    # Plot the radar chart
    ax = plt.subplot(111, polar=True)

    # Plot baseline model
    ax.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline Model', color='#3498db')
    ax.fill(angles, baseline_values, alpha=0.1, color='#3498db')

    # Plot fair model
    ax.plot(angles, fair_values, 'o-', linewidth=2, label='Fair Model', color='#2ecc71')
    ax.fill(angles, fair_values, alpha=0.1, color='#2ecc71')

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Comparison Radar Chart', fontsize=15, fontweight='bold', pad=20)

    # Save the radar chart
    plt.tight_layout()
    plt.savefig('results/figures/model_comparison_radar.png', dpi=300, bbox_inches='tight')

    # Compare model performance
    performance_comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Reward'],
        'Baseline': [baseline_evaluation['overall_accuracy'], baseline_evaluation['overall_reward']],
        'Fair': [fair_evaluation['overall_accuracy'], fair_evaluation['overall_reward']]
    })

    # Compare group-wise metrics
    baseline_group = baseline_evaluation['group_metrics']
    fair_group = fair_evaluation['group_metrics']

    # Save comparison results
    fairness_comparison.to_csv('results/fairness_comparison.csv', index=False)
    key_fairness_comparison.to_csv('results/key_fairness_comparison.csv', index=False)
    performance_comparison.to_csv('results/performance_comparison.csv', index=False)
    baseline_group.to_csv('results/baseline_group_metrics.csv', index=False)
    fair_group.to_csv('results/fair_group_metrics.csv', index=False)

    # Create comprehensive visualizations for key fairness metrics and utility
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

    # 1. Fairness Metrics Plot (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    bar_width = 0.35
    index = np.arange(len(key_metrics_names))

    # Plot fairness metrics
    bars1 = ax1.bar(index, [baseline_fairness.get(k, 0) for k in key_metrics], bar_width,
                  label='Baseline', color='blue', alpha=0.7)
    bars2 = ax1.bar(index + bar_width, [fair_fairness.get(k, 0) for k in key_metrics], bar_width,
                  label='Fair RL', color='green', alpha=0.7)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Fairness Metric')
    ax1.set_ylabel('Disparity (lower is better)')
    ax1.set_title('Comparison of Key Fairness Metrics')
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(key_metrics_names)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # 2. Utility Metrics Plot (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    utility_metrics = ['overall_accuracy', 'overall_reward']
    utility_names = ['Accuracy', 'Reward']

    # Get utility values
    baseline_utility = [baseline_evaluation[m] for m in utility_metrics]
    fair_utility = [fair_evaluation[m] for m in utility_metrics]

    # Plot utility metrics
    index2 = np.arange(len(utility_names))
    bars3 = ax2.bar(index2, baseline_utility, bar_width, label='Baseline', color='blue', alpha=0.7)
    bars4 = ax2.bar(index2 + bar_width, fair_utility, bar_width, label='Fair RL', color='green', alpha=0.7)

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel('Utility Metric')
    ax2.set_ylabel('Value (higher is better)')
    ax2.set_title('Comparison of Utility Metrics')
    ax2.set_xticks(index2 + bar_width / 2)
    ax2.set_xticklabels(utility_names)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    # 3. Group-wise Approval Rates (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])

    # Extract group data
    groups = ['Female', 'Male']
    baseline_approval = baseline_group['approval_rate'].values
    fair_approval = fair_group['approval_rate'].values

    # Set up positions
    index3 = np.arange(len(groups))

    # Plot group approval rates
    bars5 = ax3.bar(index3, baseline_approval, bar_width, label='Baseline', color='blue', alpha=0.7)
    bars6 = ax3.bar(index3 + bar_width, fair_approval, bar_width, label='Fair RL', color='green', alpha=0.7)

    # Add value labels
    for bars in [bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax3.set_xlabel('Group')
    ax3.set_ylabel('Approval Rate')
    ax3.set_title('Group-wise Approval Rates')
    ax3.set_xticks(index3 + bar_width / 2)
    ax3.set_xticklabels(groups)
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.3)

    # 4. Group-wise Accuracy (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])

    # Extract group accuracy data
    baseline_accuracy = baseline_group['accuracy'].values
    fair_accuracy = fair_group['accuracy'].values

    # Plot group accuracy
    bars7 = ax4.bar(index3, baseline_accuracy, bar_width, label='Baseline', color='blue', alpha=0.7)
    bars8 = ax4.bar(index3 + bar_width, fair_accuracy, bar_width, label='Fair RL', color='green', alpha=0.7)

    # Add value labels
    for bars in [bars7, bars8]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax4.set_xlabel('Group')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Group-wise Accuracy')
    ax4.set_xticks(index3 + bar_width / 2)
    ax4.set_xticklabels(groups)
    ax4.legend()
    ax4.grid(axis='y', linestyle='--', alpha=0.3)

    # Add overall title and adjust layout
    plt.suptitle('Fairness-Utility Trade-off Analysis', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save the comprehensive visualization
    plt.savefig('results/plots/fairness_utility_comparison.png', dpi=300)

    # Also save the original key fairness metrics plot for backward compatibility
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(key_metrics_names))

    plt.bar(index, [baseline_fairness.get(k, 0) for k in key_metrics], bar_width,
            label='Baseline', color='blue', alpha=0.7)
    plt.bar(index + bar_width, [fair_fairness.get(k, 0) for k in key_metrics], bar_width,
            label='Fair RL', color='green', alpha=0.7)

    plt.xlabel('Fairness Metric')
    plt.ylabel('Disparity (lower is better)')
    plt.title('Comparison of Key Fairness Metrics')
    plt.xticks(index + bar_width / 2, key_metrics_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/plots/key_fairness_comparison.png')

    # 13. Analyze attribution-based bias
    print("Analyzing attribution-based bias...")
    # Get attribution-based constraint if it exists
    attribution_constraint = None
    for constraint in constraint_optimizer.constraints:
        if isinstance(constraint, AttributionBasedConstraint):
            attribution_constraint = constraint
            break

    if attribution_constraint:
        # Get attribution statistics
        attr_stats = attribution_constraint.get_attribution_stats()

        # Save attribution statistics
        attr_stats_df = pd.DataFrame([attr_stats])
        attr_stats_df.to_csv('results/attribution_stats.csv', index=False)

        # Create visualization of attribution history
        if attribution_constraint.attribution_history:
            plt.figure(figsize=(10, 6))
            attributions = [entry['attribution'] for entry in attribution_constraint.attribution_history]
            plt.plot(attributions)
            plt.title('Attribution History for Sensitive Attribute')
            plt.xlabel('Training Step')
            plt.ylabel('Attribution Value')
            plt.axhline(y=0, color='r', linestyle='-')
            plt.savefig('results/plots/attribution_history.png')

            # Plot penalties
            plt.figure(figsize=(10, 6))
            plt.plot(attribution_constraint.penalty_history)
            plt.title('Penalty History for Attribution-Based Constraint')
            plt.xlabel('Training Step')
            plt.ylabel('Penalty Value')
            plt.savefig('results/plots/attribution_penalty_history.png')

    print("FairRL pipeline completed successfully!")
    print("Results saved to 'results/' directory")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='FairRL: Fair Reinforcement Learning')
    parser.add_argument('--load-models', action='store_true',
                        help='Load pre-trained models instead of training')
    parser.add_argument('--baseline-model', type=str, default='results/models/baseline_model.pth',
                        help='Path to baseline model file')
    parser.add_argument('--fair-model', type=str, default='results/models/fair_model.pth',
                        help='Path to fair model file')

    args = parser.parse_args()
    main(args)