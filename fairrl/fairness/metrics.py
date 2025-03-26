import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class FairnessMetrics:
    """
    Implementation of fairness metrics for evaluating and comparing
    bias in RL models.
    """
    
    def __init__(self):
        """Initialize fairness metrics calculator"""
        pass
    
    def demographic_parity(self, y_pred, sensitive):
        """
        Calculate demographic parity: the difference in approval rates
        between privileged and unprivileged groups
        
        Args:
            y_pred: Predicted labels (decisions)
            sensitive: Sensitive attribute values (e.g., gender)
            
        Returns:
            float: Demographic parity difference
        """
        # Get positive prediction rates for each group
        positive_rate_unprivileged = np.mean(y_pred[sensitive == 0])
        positive_rate_privileged = np.mean(y_pred[sensitive == 1])
        
        # Calculate demographic parity difference
        dp_diff = positive_rate_privileged - positive_rate_unprivileged
        
        return {
            "demographic_parity_diff": dp_diff,
            "positive_rate_unprivileged": positive_rate_unprivileged,
            "positive_rate_privileged": positive_rate_privileged
        }
    
    def equalized_odds(self, y_true, y_pred, sensitive):
        """
        Calculate equalized odds: ensuring similar true positive and false positive rates
        across sensitive groups
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive: Sensitive attribute values
            
        Returns:
            dict: Equalized odds metrics
        """
        # Unpack data by group and ground truth
        y_true_unprivileged = y_true[sensitive == 0]
        y_pred_unprivileged = y_pred[sensitive == 0]
        
        y_true_privileged = y_true[sensitive == 1]
        y_pred_privileged = y_pred[sensitive == 1]
        
        # Calculate TPR and FPR for unprivileged group
        tn_unprivileged, fp_unprivileged, fn_unprivileged, tp_unprivileged = confusion_matrix(
            y_true_unprivileged, y_pred_unprivileged, labels=[0, 1]).ravel()
        
        tpr_unprivileged = tp_unprivileged / (tp_unprivileged + fn_unprivileged) if (tp_unprivileged + fn_unprivileged) > 0 else 0
        fpr_unprivileged = fp_unprivileged / (fp_unprivileged + tn_unprivileged) if (fp_unprivileged + tn_unprivileged) > 0 else 0
        
        # Calculate TPR and FPR for privileged group
        tn_privileged, fp_privileged, fn_privileged, tp_privileged = confusion_matrix(
            y_true_privileged, y_pred_privileged, labels=[0, 1]).ravel()
        
        tpr_privileged = tp_privileged / (tp_privileged + fn_privileged) if (tp_privileged + fn_privileged) > 0 else 0
        fpr_privileged = fp_privileged / (fp_privileged + tn_privileged) if (fp_privileged + tn_privileged) > 0 else 0
        
        # Calculate differences in TPR and FPR
        tpr_diff = tpr_privileged - tpr_unprivileged
        fpr_diff = fpr_privileged - fpr_unprivileged
        
        return {
            "tpr_diff": tpr_diff,
            "fpr_diff": fpr_diff,
            "tpr_unprivileged": tpr_unprivileged,
            "tpr_privileged": tpr_privileged,
            "fpr_unprivileged": fpr_unprivileged,
            "fpr_privileged": fpr_privileged
        }
    
    def equal_opportunity(self, y_true, y_pred, sensitive):
        """
        Calculate equal opportunity: ensuring similar true positive rates
        across sensitive groups
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive: Sensitive attribute values
            
        Returns:
            dict: Equal opportunity metrics
        """
        # Use equalized odds and extract just the TPR difference
        equalized_odds_metrics = self.equalized_odds(y_true, y_pred, sensitive)
        
        return {
            "equal_opportunity_diff": equalized_odds_metrics["tpr_diff"],
            "tpr_unprivileged": equalized_odds_metrics["tpr_unprivileged"],
            "tpr_privileged": equalized_odds_metrics["tpr_privileged"]
        }
    
    def visualize_fairness_metrics(self, metrics_dict, title="Fairness Metrics"):
        """Visualize fairness metrics"""
        # Extract metrics for visualization
        metrics_to_plot = {}
        
        if "demographic_parity_diff" in metrics_dict:
            metrics_to_plot["Demographic Parity Diff"] = metrics_dict["demographic_parity_diff"]
        
        if "tpr_diff" in metrics_dict:
            metrics_to_plot["TPR Difference"] = metrics_dict["tpr_diff"]
        
        if "fpr_diff" in metrics_dict:
            metrics_to_plot["FPR Difference"] = metrics_dict["fpr_diff"]
        
        if "equal_opportunity_diff" in metrics_dict:
            metrics_to_plot["Equal Opportunity Diff"] = metrics_dict["equal_opportunity_diff"]
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(metrics_to_plot)), list(metrics_to_plot.values()), color='blue', alpha=0.7)
        
        # Add a horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add labels and values
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    0.001 + height if height >= 0 else height - 0.03,
                    f'{height:.3f}',
                    ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.xticks(range(len(metrics_to_plot)), list(metrics_to_plot.keys()), rotation=45)
        plt.title(title)
        plt.ylabel("Difference (Privileged - Unprivileged)")
        plt.tight_layout()
        
        return plt