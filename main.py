import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
from tqdm import tqdm

# Import project modules
from fairrl.data.loader import AdultDataset
from fairrl.models.environment import LoanEnvironment
from fairrl.models.models import DDQNAgent
from fairrl.explainability.ig import IGExplainer
from fairrl.fairness.metrics import FairnessMetrics
from fairrl.fairness.constraints import (
    DemographicParityConstraint, 
    EqualizedOddsConstraint,
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
        agent.step(state, action, reward, next_state, done)
        
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
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
    
    return agent, np.array(scores), pd.DataFrame(decisions)

def train_fair_model(env, agent, constraint_optimizer, num_episodes=1000, seed=42):
    """Train a fair RL model with fairness constraints"""
    set_seeds(seed)
    
    scores = []
    scores_window = deque(maxlen=100)
    decisions = []
    
    for i_episode in tqdm(range(1, num_episodes+1), desc="Training Fair Model"):
        state = env.reset()
        score = 0
        
        # Interact with environment (single step for our loan environment)
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        # Apply fairness constraints to modify reward
        sensitive = info['sensitive']
        ground_truth = info['ground_truth']
        fair_reward = constraint_optimizer.compute_fair_reward(
            state, action, reward, sensitive, ground_truth
        )
        
        # Update agent with fair reward
        agent.step(state, action, fair_reward, next_state, done)
        
        score += fair_reward
        
        # Store decision information
        decisions.append({
            'episode': i_episode,
            'action': action,
            'reward': reward,
            'fair_reward': fair_reward,
            'sensitive': sensitive,
            'ground_truth': ground_truth,
            'correct_decision': info['correct_decision']
        })
        
        # Update progress tracking
        scores_window.append(score)
        scores.append(score)
        
        # Update constraint weights
        constraint_optimizer.update_constraint_weights(i_episode)
        
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
    
    return agent, np.array(scores), pd.DataFrame(decisions)

def evaluate_model(env, agent, num_episodes=1000, seed=42):
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

def analyze_bias_with_ig(model, feature_names, dataset, fairness_metrics, num_samples=100):
    """Analyze bias using Integrated Gradients"""
    # Initialize explainer
    explainer = IGExplainer(model, feature_names)
    
    # Get samples for analysis
    # For simplicity, we'll use the first num_samples from each group
    X, y, sensitive, _ = dataset.preprocess_data()
    
    # Split by sensitive attribute
    female_indices = np.where(sensitive == 0)[0][:num_samples]
    male_indices = np.where(sensitive == 1)[0][:num_samples]
    
    female_samples = X[female_indices]
    male_samples = X[male_indices]
    
    # Compare attributions between groups
    comparison = explainer.compare_attributions(female_samples, male_samples)
    
    # Visualize attribution comparison
    plt_comparison = explainer.visualize_attribution_comparison(
        comparison, title="Female vs Male Attribution Comparison"
    )
    
    return {
        'attribution_comparison': comparison,
        'plots': {
            'attribution_comparison': plt_comparison
        }
    }

def main():
    """Main function to run the FairRL pipeline"""
    # Create directories for results
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)
    
    # 1. Load and preprocess the UCI Adult dataset
    print("Loading and preprocessing UCI Adult dataset...")
    dataset = AdultDataset()
    X, y, sensitive, preprocessor = dataset.preprocess_data()
    feature_names = dataset.get_feature_names(preprocessor)
    
    # 2. Set up RL environment
    print("Setting up RL environment...")
    env = LoanEnvironment(X, y, sensitive)
    
    # 3. Train baseline RL model
    print("Training baseline RL model...")
    state_size = X.shape[1]
    action_size = 2  # Approve or deny
    baseline_agent = DDQNAgent(state_size, action_size)
    baseline_agent, baseline_scores, baseline_decisions = train_baseline_model(
        env, baseline_agent, num_episodes=5000
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
    baseline_bias_analysis['plots']['attribution_comparison'].savefig(
        'results/plots/baseline_attribution_comparison.png'
    )
    
    # 7. Train fair RL model with constraints
    print("Training fair RL model with constraints...")
    # Create new agent and environment for fair training
    fair_env = LoanEnvironment(X, y, sensitive)
    fair_agent = DDQNAgent(state_size, action_size)
    
    # Set up fairness constraints
    demographic_parity_constraint = DemographicParityConstraint(lambda_param=0.5)
    equalized_odds_constraint = EqualizedOddsConstraint(lambda_param=0.5)
    
    # Create constraint optimizer
    constraint_optimizer = ConstraintOptimizer(
        fair_agent, [demographic_parity_constraint, equalized_odds_constraint]
    )
    
    # Train with fairness constraints
    fair_agent, fair_scores, fair_decisions = train_fair_model(
        fair_env, fair_agent, constraint_optimizer, num_episodes=5000
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
    fair_bias_analysis['plots']['attribution_comparison'].savefig(
        'results/plots/fair_attribution_comparison.png'
    )
    
    # 11. Compare baseline and fair models
    print("Comparing baseline and fair models...")
    # Compare fairness metrics
    fairness_comparison = pd.DataFrame({
        'Metric': list(baseline_fairness.keys()),
        'Baseline': list(baseline_fairness.values()),
        'Fair': list(fair_fairness.values())
    })
    
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
    performance_comparison.to_csv('results/performance_comparison.csv', index=False)
    baseline_group.to_csv('results/baseline_group_metrics.csv', index=False)
    fair_group.to_csv('results/fair_group_metrics.csv', index=False)
    
    print("FairRL pipeline completed successfully!")
    print("Results saved to 'results/' directory")

if __name__ == "__main__":
    main()