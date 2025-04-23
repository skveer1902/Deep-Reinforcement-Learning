import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set Seaborn style for all plots
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.2)

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

        # Calculate the raw difference (male - female) for reporting
        tpr_diff_raw = equalized_odds_metrics["tpr_diff"]
        # Calculate the absolute difference for the metric
        tpr_diff_abs = abs(tpr_diff_raw)

        return {
            "equal_opportunity_diff": tpr_diff_abs,  # Use absolute value for the metric
            "equal_opportunity_diff_raw": tpr_diff_raw,  # Keep raw difference for directional analysis
            "tpr_unprivileged": equalized_odds_metrics["tpr_unprivileged"],
            "tpr_privileged": equalized_odds_metrics["tpr_privileged"]
        }

    def compute_penalty(self, y_true, y_pred, sensitive, fairness_type='demographic_parity', lambda_param=0.1):
        """
        Compute fairness penalty based on specified fairness metric

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive: Sensitive attribute values
            fairness_type: Type of fairness metric to use ('demographic_parity', 'equal_opportunity')
            lambda_param: Weight of the fairness penalty

        Returns:
            float: Fairness penalty value
        """
        penalty = 0.0

        if fairness_type == 'demographic_parity':
            # Calculate demographic parity metrics
            dp_metrics = self.demographic_parity(y_pred, sensitive)
            # Penalty is proportional to the absolute difference in approval rates
            penalty = lambda_param * abs(dp_metrics['demographic_parity_diff'])

        elif fairness_type == 'equal_opportunity':
            # Calculate equal opportunity metrics
            eo_metrics = self.equal_opportunity(y_true, y_pred, sensitive)
            # Penalty is proportional to the absolute difference in true positive rates
            penalty = lambda_param * abs(eo_metrics['equal_opportunity_diff'])

        elif fairness_type == 'equalized_odds':
            # Calculate equalized odds metrics
            eodds_metrics = self.equalized_odds(y_true, y_pred, sensitive)
            # Penalty is proportional to the sum of absolute differences in TPR and FPR
            penalty = lambda_param * (abs(eodds_metrics['tpr_diff']) + abs(eodds_metrics['fpr_diff']))

        else:
            raise ValueError(f"Unknown fairness type: {fairness_type}")

        return penalty

    def compute_group_penalties(self, y_true, y_pred, sensitive, fairness_type='demographic_parity', lambda_param=0.1):
        """
        Compute group-specific penalties based on fairness violations

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive: Sensitive attribute values
            fairness_type: Type of fairness metric to use
            lambda_param: Weight of the fairness penalty

        Returns:
            dict: Group-specific penalties
        """
        penalties = {}

        if fairness_type == 'demographic_parity':
            # Calculate demographic parity metrics
            dp_metrics = self.demographic_parity(y_pred, sensitive)
            dp_diff = dp_metrics['demographic_parity_diff']

            # If privileged group has higher approval rate, penalize approving privileged
            # and denying unprivileged
            if dp_diff > 0:
                penalties['privileged_approve'] = lambda_param * dp_diff
                penalties['unprivileged_deny'] = lambda_param * dp_diff
                penalties['privileged_deny'] = 0
                penalties['unprivileged_approve'] = 0
            else:
                # If unprivileged group has higher approval rate, penalize approving unprivileged
                # and denying privileged
                penalties['privileged_deny'] = lambda_param * abs(dp_diff)
                penalties['unprivileged_approve'] = lambda_param * abs(dp_diff)
                penalties['privileged_approve'] = 0
                penalties['unprivileged_deny'] = 0

        elif fairness_type == 'equal_opportunity':
            # Calculate equal opportunity metrics
            eo_metrics = self.equal_opportunity(y_true, y_pred, sensitive)
            tpr_diff = eo_metrics['equal_opportunity_diff']

            # Only apply penalties to positive ground truth cases (y_true == 1)
            if tpr_diff > 0:
                # If privileged group has higher TPR, penalize denying qualified unprivileged
                penalties['unprivileged_qualified_deny'] = lambda_param * tpr_diff
                penalties['privileged_qualified_deny'] = 0
            else:
                # If unprivileged group has higher TPR, penalize denying qualified privileged
                penalties['privileged_qualified_deny'] = lambda_param * abs(tpr_diff)
                penalties['unprivileged_qualified_deny'] = 0

        elif fairness_type == 'equalized_odds':
            # Calculate equalized odds metrics
            eodds_metrics = self.equalized_odds(y_true, y_pred, sensitive)
            tpr_diff = eodds_metrics['tpr_diff']
            fpr_diff = eodds_metrics['fpr_diff']

            # TPR penalties (for y_true == 1)
            if tpr_diff > 0:
                penalties['unprivileged_qualified_deny'] = lambda_param * tpr_diff
                penalties['privileged_qualified_deny'] = 0
            else:
                penalties['privileged_qualified_deny'] = lambda_param * abs(tpr_diff)
                penalties['unprivileged_qualified_deny'] = 0

            # FPR penalties (for y_true == 0)
            if fpr_diff > 0:
                penalties['privileged_unqualified_approve'] = lambda_param * fpr_diff
                penalties['unprivileged_unqualified_approve'] = 0
            else:
                penalties['unprivileged_unqualified_approve'] = lambda_param * abs(fpr_diff)
                penalties['privileged_unqualified_approve'] = 0

        else:
            raise ValueError(f"Unknown fairness type: {fairness_type}")

        return penalties

    def visualize_fairness_metrics(self, metrics_dict, title="Fairness Metrics"):
        """Visualize fairness metrics with enhanced styling using Seaborn"""
        # Extract metrics for visualization
        metrics_data = []

        # Add demographic parity metrics
        if "demographic_parity_diff" in metrics_dict:
            metrics_data.append({
                'Metric': 'Demographic Parity',
                'Value': metrics_dict["demographic_parity_diff"],
                'Type': 'Difference',
                'Description': 'Difference in approval rates'
            })

            if "positive_rate_privileged" in metrics_dict and "positive_rate_unprivileged" in metrics_dict:
                metrics_data.append({
                    'Metric': 'Approval Rate (Male)',
                    'Value': metrics_dict["positive_rate_privileged"],
                    'Type': 'Rate',
                    'Description': 'Approval rate for males'
                })
                metrics_data.append({
                    'Metric': 'Approval Rate (Female)',
                    'Value': metrics_dict["positive_rate_unprivileged"],
                    'Type': 'Rate',
                    'Description': 'Approval rate for females'
                })

        # Add equal opportunity metrics
        if "equal_opportunity_diff" in metrics_dict:
            metrics_data.append({
                'Metric': 'Equal Opportunity',
                'Value': metrics_dict["equal_opportunity_diff"],
                'Type': 'Difference',
                'Description': 'Difference in true positive rates'
            })

            if "tpr_privileged" in metrics_dict and "tpr_unprivileged" in metrics_dict:
                metrics_data.append({
                    'Metric': 'TPR (Male)',
                    'Value': metrics_dict["tpr_privileged"],
                    'Type': 'Rate',
                    'Description': 'True positive rate for males'
                })
                metrics_data.append({
                    'Metric': 'TPR (Female)',
                    'Value': metrics_dict["tpr_unprivileged"],
                    'Type': 'Rate',
                    'Description': 'True positive rate for females'
                })

        # Add equalized odds metrics
        if "tpr_diff" in metrics_dict and "fpr_diff" in metrics_dict:
            metrics_data.append({
                'Metric': 'TPR Difference',
                'Value': metrics_dict["tpr_diff"],
                'Type': 'Difference',
                'Description': 'Difference in true positive rates'
            })
            metrics_data.append({
                'Metric': 'FPR Difference',
                'Value': metrics_dict["fpr_diff"],
                'Type': 'Difference',
                'Description': 'Difference in false positive rates'
            })

            if "fpr_privileged" in metrics_dict and "fpr_unprivileged" in metrics_dict:
                metrics_data.append({
                    'Metric': 'FPR (Male)',
                    'Value': metrics_dict["fpr_privileged"],
                    'Type': 'Rate',
                    'Description': 'False positive rate for males'
                })
                metrics_data.append({
                    'Metric': 'FPR (Female)',
                    'Value': metrics_dict["fpr_unprivileged"],
                    'Type': 'Rate',
                    'Description': 'False positive rate for females'
                })

        # Create DataFrame for plotting
        df = pd.DataFrame(metrics_data)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot differences in the first subplot
        diff_df = df[df['Type'] == 'Difference']
        if not diff_df.empty:
            # Use a diverging color palette for differences
            colors = ['#e74c3c' if val > 0 else '#2ecc71' for val in diff_df['Value']]

            # Create the bar plot
            sns.barplot(x='Metric', y='Value', data=diff_df, ax=ax1, palette=colors)

            # Add a horizontal line at y=0
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Add value labels
            for i, p in enumerate(ax1.patches):
                height = p.get_height()
                if np.isnan(height):
                    continue

                ax1.text(p.get_x() + p.get_width()/2.,
                        0.001 + height if height >= 0 else height - 0.03,
                        f'{height:.3f}',
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontweight='bold', fontsize=10)

            # Customize the plot
            ax1.set_title('Fairness Differences', fontsize=14, fontweight='bold', pad=15)
            ax1.set_xlabel('')
            ax1.set_ylabel('Difference (Male - Female)', fontsize=12, labelpad=10)
            ax1.tick_params(axis='x', rotation=45)

        # Plot rates in the second subplot
        rate_df = df[df['Type'] == 'Rate']
        if not rate_df.empty:
            # Create the bar plot with a categorical hue
            sns.barplot(x='Metric', y='Value', data=rate_df, ax=ax2,
                       palette=sns.color_palette('viridis', n_colors=len(rate_df)))

            # Add value labels
            for i, p in enumerate(ax2.patches):
                height = p.get_height()
                if np.isnan(height):
                    continue

                ax2.text(p.get_x() + p.get_width()/2.,
                        height + 0.01,
                        f'{height:.3f}',
                        ha='center', va='bottom',
                        fontweight='bold', fontsize=10)

            # Customize the plot
            ax2.set_title('Group Rates', fontsize=14, fontweight='bold', pad=15)
            ax2.set_xlabel('')
            ax2.set_ylabel('Rate', fontsize=12, labelpad=10)
            ax2.tick_params(axis='x', rotation=45)

        # Add a title to the entire figure
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title

        return fig