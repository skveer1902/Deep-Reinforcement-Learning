import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fairrl.explainability.ig import IGExplainer
from fairrl.fairness.metrics import FairnessMetrics

class FairnessConstraint:
    """
    Base class for fairness constraints that can be used
    to modify the reward function in RL.
    """

    def __init__(self, lambda_param=0.1):
        """
        Initialize fairness constraint

        Args:
            lambda_param: Weight of the fairness constraint
        """
        self.lambda_param = lambda_param

    def compute_penalty(self, state, action, sensitive):
        """
        Compute fairness penalty to be applied to reward

        Args:
            state: Current state
            action: Chosen action
            sensitive: Sensitive attribute value

        Returns:
            float: Fairness penalty
        """
        raise NotImplementedError("Subclasses must implement this method")

class DemographicParityConstraint(FairnessConstraint):
    """
    Implement demographic parity constraint:
    P(approve | female) should be similar to P(approve | male)
    """

    def __init__(self, lambda_param=0.1):
        super().__init__(lambda_param)
        self.approve_female_count = 0
        self.female_count = 0
        self.approve_male_count = 0
        self.male_count = 0
        self.fairness_metrics = FairnessMetrics()
        self.decisions = []

    def update_counts(self, action, sensitive):
        """Update approval and total counts by group"""
        if sensitive == 0:  # Female
            self.female_count += 1
            if action == 1:  # Approve
                self.approve_female_count += 1
        else:  # Male
            self.male_count += 1
            if action == 1:  # Approve
                self.approve_male_count += 1

        # Store decision for batch analysis
        self.decisions.append({
            'action': action,
            'sensitive': sensitive
        })

        # Keep decision history manageable
        if len(self.decisions) > 1000:
            self.decisions = self.decisions[-1000:]

    def compute_penalty(self, state, action, sensitive):
        """Compute demographic parity penalty with targeted effectiveness for better utility"""
        self.update_counts(action, sensitive)

        # Calculate approval rates
        female_rate = self.approve_female_count / max(1, self.female_count)
        male_rate = self.approve_male_count / max(1, self.male_count)

        # Calculate disparity
        disparity = abs(female_rate - male_rate)

        # Apply penalty when taking action that increases disparity
        penalty = 0

        # Linear scaling for disparity to be less aggressive overall but still effective
        # This helps maintain better utility while still improving demographic parity
        disparity_scaled = disparity * 10.0  # Less aggressive than quadratic scaling

        # More targeted penalty calculation for demographic parity
        if sensitive == 0:  # Female
            if action == 0 and female_rate < male_rate:
                # Denying a female when female approval rate is already lower
                # This directly increases demographic parity disparity
                penalty = self.lambda_param * disparity_scaled * 3.0
            elif action == 1 and female_rate > male_rate:
                # Approving a female when female approval rate is already higher
                # Less aggressive penalty to maintain utility
                penalty = self.lambda_param * disparity_scaled * 0.5
        else:  # Male
            if action == 1 and male_rate > female_rate:
                # Approving a male when male approval rate is already higher
                # This directly increases demographic parity disparity
                penalty = self.lambda_param * disparity_scaled * 3.0
            elif action == 0 and male_rate < female_rate:
                # Denying a male when male approval rate is already lower
                # Less aggressive penalty to maintain utility
                penalty = self.lambda_param * disparity_scaled * 0.5

        # Add a moderate base penalty proportional to the current disparity
        # Less aggressive to maintain better utility
        base_penalty = self.lambda_param * disparity * 2.0  # Linear scaling for base penalty
        penalty += base_penalty

        # Add a strong reward component for actions that reduce disparity
        if sensitive == 0:  # Female
            if action == 1 and female_rate < male_rate:
                # Approving a female when female approval rate is lower (reduces disparity)
                return -self.lambda_param * 8.0  # Stronger reward for demographic parity improvement
            elif action == 0 and female_rate > male_rate:
                # Denying a female when female approval rate is higher (reduces disparity)
                return -self.lambda_param * 4.0  # Moderate reward
        else:  # Male
            if action == 0 and male_rate > female_rate:
                # Denying a male when male approval rate is higher (reduces disparity)
                return -self.lambda_param * 8.0  # Stronger reward for demographic parity improvement
            elif action == 1 and male_rate < female_rate:
                # Approving a male when male approval rate is lower (reduces disparity)
                return -self.lambda_param * 4.0  # Moderate reward

        # Lower penalty cap to maintain better utility
        return min(penalty, 20.0)

    def get_metrics(self):
        """Get current demographic parity metrics"""
        if not self.decisions:
            return {}

        # Extract actions and sensitive attributes
        y_pred = np.array([d['action'] for d in self.decisions])
        sensitive = np.array([d['sensitive'] for d in self.decisions])

        # Calculate demographic parity metrics
        return self.fairness_metrics.demographic_parity(y_pred, sensitive)

class EqualOpportunityConstraint(FairnessConstraint):
    """
    Implement equal opportunity constraint:
    True positive rates should be similar across groups
    """

    def __init__(self, lambda_param=0.1):
        super().__init__(lambda_param)

        # Counters for female
        self.tp_female = 0  # True positives
        self.pos_female = 0  # Actual positive cases

        # Counters for male
        self.tp_male = 0
        self.pos_male = 0

        # For metrics calculation
        self.fairness_metrics = FairnessMetrics()
        self.decisions = []

    def update_counts(self, action, ground_truth, sensitive):
        """Update counts based on decision outcomes"""
        # Only update counts for positive ground truth cases (qualified applicants)
        if ground_truth == 1:
            if sensitive == 0:  # Female
                self.pos_female += 1
                if action == 1:  # Approve
                    self.tp_female += 1
            else:  # Male
                self.pos_male += 1
                if action == 1:  # Approve
                    self.tp_male += 1

        # Store decision for batch analysis
        self.decisions.append({
            'action': action,
            'ground_truth': ground_truth,
            'sensitive': sensitive
        })

        # Keep decision history manageable
        if len(self.decisions) > 1000:
            self.decisions = self.decisions[-1000:]

    def compute_penalty(self, state, action, sensitive, ground_truth):
        """Compute equal opportunity penalty with balanced effectiveness"""
        self.update_counts(action, ground_truth, sensitive)

        # Only apply penalties to positive ground truth cases (qualified applicants)
        if ground_truth != 1:
            return 0

        # Calculate TPR for both groups
        tpr_female = self.tp_female / max(1, self.pos_female)
        tpr_male = self.tp_male / max(1, self.pos_male)

        # Calculate TPR disparity
        tpr_disparity = abs(tpr_female - tpr_male)

        # Moderate scaling for disparity - balanced approach
        tpr_disparity_scaled = tpr_disparity * 8.0  # More aggressive than before but still balanced

        # Compute penalty based on the action and current disparities
        penalty = 0

        # More effective penalty calculation for equal opportunity
        if sensitive == 0:  # Female
            if action == 0 and tpr_female < tpr_male:
                # Denying a qualified female when female TPR is already lower
                # This directly increases equal opportunity disparity
                penalty = self.lambda_param * tpr_disparity_scaled * 3.0  # Stronger penalty
            elif action == 1 and tpr_female > tpr_male:
                # Approving a qualified female when female TPR is already higher
                penalty = self.lambda_param * tpr_disparity_scaled * 1.0  # Moderate penalty
        else:  # Male
            if action == 0 and tpr_male < tpr_female:
                # Denying a qualified male when male TPR is already lower
                # This directly increases equal opportunity disparity
                penalty = self.lambda_param * tpr_disparity_scaled * 3.0  # Stronger penalty
            elif action == 1 and tpr_male > tpr_female:
                # Approving a qualified male when male TPR is already higher
                penalty = self.lambda_param * tpr_disparity_scaled * 1.0  # Moderate penalty

        # Add a moderate base penalty proportional to the current disparity
        base_penalty = self.lambda_param * tpr_disparity * 2.0  # Moderate base penalty
        penalty += base_penalty

        # Add a stronger reward component for actions that reduce disparity
        if sensitive == 0:  # Female
            if action == 1 and tpr_female < tpr_male:
                # Approving a qualified female when female TPR is lower (reduces disparity)
                return -self.lambda_param * 5.0  # Stronger reward for equal opportunity improvement
            elif action == 0 and tpr_female > tpr_male:
                # Denying a qualified female when female TPR is higher (reduces disparity)
                return -self.lambda_param * 2.0  # Moderate reward
        else:  # Male
            if action == 0 and tpr_male > tpr_female:
                # Denying a qualified male when male TPR is higher (reduces disparity)
                return -self.lambda_param * 5.0  # Stronger reward for equal opportunity improvement
            elif action == 1 and tpr_male < tpr_female:
                # Approving a qualified male when male TPR is lower (reduces disparity)
                return -self.lambda_param * 2.0  # Moderate reward

        # Moderate penalty cap to balance fairness and utility
        return min(penalty, 15.0)

    def get_metrics(self):
        """Get current equal opportunity metrics"""
        if not self.decisions:
            return {}

        # Extract actions, ground truth, and sensitive attributes
        y_pred = np.array([d['action'] for d in self.decisions])
        y_true = np.array([d['ground_truth'] for d in self.decisions])
        sensitive = np.array([d['sensitive'] for d in self.decisions])

        # Calculate equal opportunity metrics
        return self.fairness_metrics.equal_opportunity(y_true, y_pred, sensitive)

class EqualizedOddsConstraint(FairnessConstraint):
    """
    Implement equalized odds constraint:
    True positive and false positive rates should be similar across groups
    """

    def __init__(self, lambda_param=0.1):
        super().__init__(lambda_param)

        # Counters for female
        self.tp_female = 0  # True positives
        self.fp_female = 0  # False positives
        self.pos_female = 0  # Actual positive cases
        self.neg_female = 0  # Actual negative cases

        # Counters for male
        self.tp_male = 0
        self.fp_male = 0
        self.pos_male = 0
        self.neg_male = 0

        # For metrics calculation
        self.fairness_metrics = FairnessMetrics()
        self.decisions = []

    def update_counts(self, action, ground_truth, sensitive):
        """Update counts based on decision outcomes"""
        if sensitive == 0:  # Female
            if ground_truth == 1:
                self.pos_female += 1
                if action == 1:
                    self.tp_female += 1
            else:
                self.neg_female += 1
                if action == 1:
                    self.fp_female += 1
        else:  # Male
            if ground_truth == 1:
                self.pos_male += 1
                if action == 1:
                    self.tp_male += 1
            else:
                self.neg_male += 1
                if action == 1:
                    self.fp_male += 1

        # Store decision for batch analysis
        self.decisions.append({
            'action': action,
            'ground_truth': ground_truth,
            'sensitive': sensitive
        })

        # Keep decision history manageable
        if len(self.decisions) > 1000:
            self.decisions = self.decisions[-1000:]

    def compute_penalty(self, state, action, sensitive, ground_truth):
        """Compute equalized odds penalty"""
        self.update_counts(action, ground_truth, sensitive)

        # Calculate TPR and FPR for both groups
        tpr_female = self.tp_female / max(1, self.pos_female)
        fpr_female = self.fp_female / max(1, self.neg_female)

        tpr_male = self.tp_male / max(1, self.pos_male)
        fpr_male = self.fp_male / max(1, self.neg_male)

        # Calculate disparities
        tpr_disparity = abs(tpr_female - tpr_male)
        fpr_disparity = abs(fpr_female - fpr_male)
        total_disparity = tpr_disparity + fpr_disparity

        # Compute penalty based on the action and current disparities
        penalty = 0

        if sensitive == 0:  # Female
            if ground_truth == 1 and action == 0 and tpr_female < tpr_male:
                # Denying a qualified female when female TPR is already lower
                penalty = self.lambda_param * tpr_disparity
            elif ground_truth == 0 and action == 1 and fpr_female > fpr_male:
                # Approving an unqualified female when female FPR is already higher
                penalty = self.lambda_param * fpr_disparity
        else:  # Male
            if ground_truth == 1 and action == 0 and tpr_male < tpr_female:
                # Denying a qualified male when male TPR is already lower
                penalty = self.lambda_param * tpr_disparity
            elif ground_truth == 0 and action == 1 and fpr_male > fpr_female:
                # Approving an unqualified male when male FPR is already higher
                penalty = self.lambda_param * fpr_disparity

        return penalty

    def get_metrics(self):
        """Get current equalized odds metrics"""
        if not self.decisions:
            return {}

        # Extract actions, ground truth, and sensitive attributes
        y_pred = np.array([d['action'] for d in self.decisions])
        y_true = np.array([d['ground_truth'] for d in self.decisions])
        sensitive = np.array([d['sensitive'] for d in self.decisions])

        # Calculate equalized odds metrics
        return self.fairness_metrics.equalized_odds(y_true, y_pred, sensitive)

class ConstraintOptimizer:
    """
    Optimizer that adjusts RL training to satisfy fairness constraints
    """

    def __init__(self, agent, constraints=None, lambda_scheduler=None):
        """
        Initialize constraint optimizer

        Args:
            agent: The RL agent to optimize
            constraints: List of fairness constraints
            lambda_scheduler: Scheduler for constraint weights
        """
        self.agent = agent
        self.constraints = constraints or []
        self.lambda_scheduler = lambda_scheduler

    def add_constraint(self, constraint):
        """Add a fairness constraint"""
        self.constraints.append(constraint)

    def compute_fair_reward(self, state, action, reward, sensitive, ground_truth=None):
        """
        Apply fairness constraints to modify the reward with stability safeguards

        Args:
            state: Current state
            action: Chosen action
            reward: Original reward
            sensitive: Sensitive attribute value
            ground_truth: Actual outcome (for some constraints)

        Returns:
            float: Modified reward with fairness penalties
        """
        fair_reward = reward
        total_penalty = 0

        for constraint in self.constraints:
            try:
                # Compute penalty based on constraint type
                if isinstance(constraint, (EqualizedOddsConstraint, EqualOpportunityConstraint)) and ground_truth is not None:
                    penalty = constraint.compute_penalty(state, action, sensitive, ground_truth)
                elif isinstance(constraint, AttributionBasedConstraint) and ground_truth is not None:
                    penalty = constraint.compute_penalty(state, action, sensitive, ground_truth)
                else:
                    penalty = constraint.compute_penalty(state, action, sensitive)

                # Ensure penalty is finite and within reasonable bounds
                if not np.isfinite(penalty):
                    # Skip this penalty if it's not finite
                    continue

                # Add to total penalty (will be applied at the end)
                total_penalty += penalty

            except TypeError as e:
                # Handle the case where a constraint requires ground_truth but it's not provided
                if 'ground_truth' in str(e) and ground_truth is None:
                    # Skip this constraint if ground_truth is required but not available
                    continue
                else:
                    # Re-raise other TypeError exceptions
                    raise
            except Exception as e:
                # Catch any other exceptions to prevent training failure
                continue

        # Apply total penalty with a much higher cap to allow for extremely strong fairness enforcement
        # This ensures that fairness becomes the dominant factor in the reward signal when violations occur
        capped_penalty = np.clip(total_penalty, -10, 100)  # Much higher cap and allow negative penalties (rewards)
        fair_reward -= capped_penalty

        # Add a direct fairness reward component that rewards fair decisions
        # This is separate from the constraint penalties and provides a positive signal for fair behavior
        if total_penalty < 0:  # Negative penalty means the action improved fairness
            fair_reward += 2.0  # Add a significant bonus for fairness-improving actions

        # Final safety check to ensure reward is finite
        if not np.isfinite(fair_reward):
            return reward

        return fair_reward

    def update_constraint_weights(self, iteration):
        """Update constraint weights using scheduler (if provided)"""
        if self.lambda_scheduler:
            for constraint in self.constraints:
                constraint.lambda_param = self.lambda_scheduler(constraint.lambda_param, iteration)

        # Update adaptive lambda for attribution-based constraints
        for constraint in self.constraints:
            if isinstance(constraint, AttributionBasedConstraint):
                constraint.update_adaptive_lambda()

    def get_constraint_stats(self):
        """Get statistics about constraints"""
        stats = {}

        for i, constraint in enumerate(self.constraints):
            constraint_type = type(constraint).__name__

            if isinstance(constraint, AttributionBasedConstraint):
                stats[f"{constraint_type}_{i}"] = constraint.get_attribution_stats()
            elif isinstance(constraint, DemographicParityConstraint):
                # Use the get_metrics method if available
                if hasattr(constraint, 'get_metrics') and callable(constraint.get_metrics):
                    metrics = constraint.get_metrics()
                    if metrics:
                        metrics['lambda'] = constraint.lambda_param
                        stats[f"{constraint_type}_{i}"] = metrics
                    else:
                        # Fallback to direct calculation
                        female_rate = constraint.approve_female_count / max(1, constraint.female_count)
                        male_rate = constraint.approve_male_count / max(1, constraint.male_count)

                        stats[f"{constraint_type}_{i}"] = {
                            "female_approval_rate": female_rate,
                            "male_approval_rate": male_rate,
                            "disparity": abs(female_rate - male_rate),
                            "lambda": constraint.lambda_param
                        }
                else:
                    # Fallback to direct calculation
                    female_rate = constraint.approve_female_count / max(1, constraint.female_count)
                    male_rate = constraint.approve_male_count / max(1, constraint.male_count)

                    stats[f"{constraint_type}_{i}"] = {
                        "female_approval_rate": female_rate,
                        "male_approval_rate": male_rate,
                        "disparity": abs(female_rate - male_rate),
                        "lambda": constraint.lambda_param
                    }
            elif isinstance(constraint, EqualOpportunityConstraint):
                # Use the get_metrics method if available
                if hasattr(constraint, 'get_metrics') and callable(constraint.get_metrics):
                    metrics = constraint.get_metrics()
                    if metrics:
                        metrics['lambda'] = constraint.lambda_param
                        stats[f"{constraint_type}_{i}"] = metrics
                    else:
                        # Fallback to direct calculation
                        tpr_female = constraint.tp_female / max(1, constraint.pos_female)
                        tpr_male = constraint.tp_male / max(1, constraint.pos_male)

                        stats[f"{constraint_type}_{i}"] = {
                            "tpr_female": tpr_female,
                            "tpr_male": tpr_male,
                            "tpr_disparity": abs(tpr_female - tpr_male),
                            "lambda": constraint.lambda_param
                        }
                else:
                    # Fallback to direct calculation
                    tpr_female = constraint.tp_female / max(1, constraint.pos_female)
                    tpr_male = constraint.tp_male / max(1, constraint.pos_male)

                    stats[f"{constraint_type}_{i}"] = {
                        "tpr_female": tpr_female,
                        "tpr_male": tpr_male,
                        "tpr_disparity": abs(tpr_female - tpr_male),
                        "lambda": constraint.lambda_param
                    }
            elif isinstance(constraint, EqualizedOddsConstraint):
                # Use the get_metrics method if available
                if hasattr(constraint, 'get_metrics') and callable(constraint.get_metrics):
                    metrics = constraint.get_metrics()
                    if metrics:
                        metrics['lambda'] = constraint.lambda_param
                        stats[f"{constraint_type}_{i}"] = metrics
                    else:
                        # Fallback to direct calculation
                        tpr_female = constraint.tp_female / max(1, constraint.pos_female)
                        fpr_female = constraint.fp_female / max(1, constraint.neg_female)
                        tpr_male = constraint.tp_male / max(1, constraint.pos_male)
                        fpr_male = constraint.fp_male / max(1, constraint.neg_male)

                        stats[f"{constraint_type}_{i}"] = {
                            "tpr_female": tpr_female,
                            "fpr_female": fpr_female,
                            "tpr_male": tpr_male,
                            "fpr_male": fpr_male,
                            "tpr_disparity": abs(tpr_female - tpr_male),
                            "fpr_disparity": abs(fpr_female - fpr_male),
                            "lambda": constraint.lambda_param
                        }
                else:
                    # Fallback to direct calculation
                    tpr_female = constraint.tp_female / max(1, constraint.pos_female)
                    fpr_female = constraint.fp_female / max(1, constraint.neg_female)
                    tpr_male = constraint.tp_male / max(1, constraint.pos_male)
                    fpr_male = constraint.fp_male / max(1, constraint.neg_male)

                    stats[f"{constraint_type}_{i}"] = {
                        "tpr_female": tpr_female,
                        "fpr_female": fpr_female,
                        "tpr_male": tpr_male,
                        "fpr_male": fpr_male,
                        "tpr_disparity": abs(tpr_female - tpr_male),
                        "fpr_disparity": abs(fpr_female - fpr_male),
                        "lambda": constraint.lambda_param
                    }

        return stats

class AttributionBasedConstraint(FairnessConstraint):
    """
    Constraint that uses feature attributions to detect and mitigate bias.
    This constraint dynamically adjusts penalties based on how much the model
    relies on sensitive attributes for decision making.
    """

    def __init__(self, model, sensitive_attr_idx, lambda_param=0.1, attribution_threshold=0.1):
        """
        Initialize attribution-based constraint

        Args:
            model: The RL model (Q-Network)
            sensitive_attr_idx: Index of the sensitive attribute
            lambda_param: Weight of the fairness constraint
            attribution_threshold: Threshold for determining significant attributions
        """
        super().__init__(lambda_param)
        self.sensitive_attr_idx = sensitive_attr_idx
        self.attribution_threshold = attribution_threshold
        self.explainer = IGExplainer(model)

        # Track attributions for analysis
        self.attribution_history = []
        self.penalty_history = []

        # Adaptive lambda parameters with higher values
        self.adaptive_lambda = lambda_param
        self.min_lambda = 0.1  # Higher minimum
        self.max_lambda = 3.0  # Higher maximum

    def compute_penalty(self, state, action, sensitive, ground_truth=None):
        """
        Compute penalty based on feature attributions with enhanced effectiveness and stability

        Args:
            state: Current state
            action: Chosen action
            sensitive: Sensitive attribute value
            ground_truth: Actual outcome (optional)

        Returns:
            float: Fairness penalty
        """
        # Convert action to int if it's a numpy type
        if isinstance(action, (np.int32, np.int64)):
            action = int(action)

        # Get attributions for this state-action pair
        attributions, _ = self.explainer.explain_decision(state, action)

        # Get attribution for sensitive attribute
        sensitive_attr_attribution = attributions[self.sensitive_attr_idx]

        # Store attribution for analysis
        self.attribution_history.append({
            'attribution': sensitive_attr_attribution,
            'sensitive': sensitive,
            'action': action,
            'ground_truth': ground_truth
        })

        # Enhanced penalty calculation with stronger scaling for better fairness
        # Use a non-linear scaling for more aggressive bias mitigation
        base_penalty = self.adaptive_lambda * min(abs(sensitive_attr_attribution) * 4.0, 2.0)

        # Apply additional penalty if attribution exceeds threshold, with stronger effect
        if abs(sensitive_attr_attribution) > self.attribution_threshold:
            # Non-linear penalty for attributions that exceed the threshold
            excess = min(abs(sensitive_attr_attribution) - self.attribution_threshold, 0.5)
            # Use quadratic scaling for stronger effect
            additional_penalty = self.adaptive_lambda * (excess ** 2) * 8.0
            penalty = base_penalty + additional_penalty
        else:
            penalty = base_penalty

        # Apply stronger context-specific penalties
        if sensitive == 0:  # Female
            if action == 0 and sensitive_attr_attribution < 0:
                # Denying a female with negative attribution to sensitive attribute
                # This suggests the model is using gender to deny females
                penalty = min(penalty * 3.0, 8.0)  # Much stronger penalty
            # Add a small penalty for any decision that uses the sensitive attribute
            elif abs(sensitive_attr_attribution) > self.attribution_threshold * 0.5:
                penalty += self.adaptive_lambda * 0.5
        else:  # Male
            if action == 1 and sensitive_attr_attribution > 0:
                # Approving a male with positive attribution to sensitive attribute
                # This suggests the model is using gender to approve males
                penalty = min(penalty * 3.0, 8.0)  # Much stronger penalty
            # Add a small penalty for any decision that uses the sensitive attribute
            elif abs(sensitive_attr_attribution) > self.attribution_threshold * 0.5:
                penalty += self.adaptive_lambda * 0.5

        # Store penalty for analysis
        self.penalty_history.append(penalty)

        # Ensure penalty is bounded to prevent numerical instability, but with a higher cap
        return min(penalty, 10.0)

    def update_adaptive_lambda(self, recent_window=100):
        """
        Update adaptive lambda parameter based on recent attributions with stability controls

        Args:
            recent_window: Number of recent samples to consider
        """
        if len(self.attribution_history) < 10:  # Need some history to adapt
            return

        # Get recent attributions
        recent_attributions = [abs(entry['attribution']) for entry in self.attribution_history[-recent_window:]]
        recent_sensitive = [entry['sensitive'] for entry in self.attribution_history[-recent_window:]]
        recent_actions = [entry['action'] for entry in self.attribution_history[-recent_window:]]

        # Compute average attribution magnitude
        avg_attribution = np.mean(recent_attributions)

        # Compute group-specific approval rates
        female_indices = [i for i, s in enumerate(recent_sensitive) if s == 0]
        male_indices = [i for i, s in enumerate(recent_sensitive) if s == 1]

        female_approvals = [recent_actions[i] for i in female_indices] if female_indices else []
        male_approvals = [recent_actions[i] for i in male_indices] if male_indices else []

        female_approval_rate = np.mean(female_approvals) if female_approvals else 0
        male_approval_rate = np.mean(male_approvals) if male_approvals else 0

        # Calculate demographic parity disparity
        dp_disparity = abs(female_approval_rate - male_approval_rate)

        # Controlled adaptive lambda adjustment with stability limits
        if avg_attribution > self.attribution_threshold or dp_disparity > 0.05:
            # Moderate increase when both attribution and disparity are high
            if avg_attribution > self.attribution_threshold and dp_disparity > 0.1:
                # Capped increase for severe cases
                increase_factor = min(1.1 + dp_disparity * 0.5, 1.2)
                self.adaptive_lambda = min(self.adaptive_lambda * increase_factor, self.max_lambda)
            else:
                # Small linear increase for moderate cases
                self.adaptive_lambda = min(self.adaptive_lambda * 1.05, self.max_lambda)
        else:
            # Gradual decrease when both attribution and disparity are low
            self.adaptive_lambda = max(self.adaptive_lambda * 0.98, self.min_lambda)

        # Update max_lambda based on observed disparity with higher caps
        # For severe disparities, allow much higher maximum lambda
        if dp_disparity > 0.2:
            self.max_lambda = 5.0  # Much higher for severe disparities
        elif dp_disparity > 0.1:
            self.max_lambda = 3.0  # Higher for moderate disparities
        else:
            self.max_lambda = 2.0  # Higher baseline

    def get_attribution_stats(self):
        """
        Get statistics about attributions

        Returns:
            dict: Attribution statistics
        """
        if not self.attribution_history:
            return {}

        # Extract attributions
        attributions = [entry['attribution'] for entry in self.attribution_history]

        # Compute statistics
        mean_attribution = np.mean(attributions)
        std_attribution = np.std(attributions)
        abs_mean_attribution = np.mean(np.abs(attributions))
        max_attribution = np.max(np.abs(attributions))

        # Group by sensitive attribute
        female_attributions = [entry['attribution'] for entry in self.attribution_history
                              if entry['sensitive'] == 0]
        male_attributions = [entry['attribution'] for entry in self.attribution_history
                            if entry['sensitive'] == 1]

        # Compute group statistics if we have data
        female_mean = np.mean(female_attributions) if female_attributions else 0
        male_mean = np.mean(male_attributions) if male_attributions else 0

        return {
            'mean_attribution': mean_attribution,
            'std_attribution': std_attribution,
            'abs_mean_attribution': abs_mean_attribution,
            'max_attribution': max_attribution,
            'female_mean_attribution': female_mean,
            'male_mean_attribution': male_mean,
            'attribution_difference': female_mean - male_mean,
            'current_lambda': self.adaptive_lambda
        }