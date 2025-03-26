import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
    
    def compute_penalty(self, state, action, sensitive):
        """Compute demographic parity penalty"""
        self.update_counts(action, sensitive)
        
        # Calculate approval rates
        female_rate = self.approve_female_count / max(1, self.female_count)
        male_rate = self.approve_male_count / max(1, self.male_count)
        
        # Calculate disparity
        disparity = abs(female_rate - male_rate)
        
        # Apply penalty when taking action that increases disparity
        penalty = 0
        if sensitive == 0 and action == 0 and female_rate < male_rate:
            # Denying a female when female approval rate is already lower
            penalty = self.lambda_param * disparity
        elif sensitive == 1 and action == 1 and male_rate > female_rate:
            # Approving a male when male approval rate is already higher
            penalty = self.lambda_param * disparity
        
        return penalty

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
        Apply fairness constraints to modify the reward
        
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
        
        for constraint in self.constraints:
            if isinstance(constraint, EqualizedOddsConstraint) and ground_truth is not None:
                penalty = constraint.compute_penalty(state, action, sensitive, ground_truth)
            else:
                penalty = constraint.compute_penalty(state, action, sensitive)
            fair_reward -= penalty
        
        return fair_reward
    
    def update_constraint_weights(self, iteration):
        """Update constraint weights using scheduler (if provided)"""
        if self.lambda_scheduler:
            for constraint in self.constraints:
                constraint.lambda_param = self.lambda_scheduler(constraint.lambda_param, iteration)