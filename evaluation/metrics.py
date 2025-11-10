"""
Comprehensive Model Evaluation Module

This module provides detailed model evaluation including performance metrics,
bias analysis, confusion matrix analysis, and model interpretability reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report,
                           precision_recall_curve, roc_curve)
from sklearn.calibration import calibration_curve
import logging
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    A comprehensive evaluator for healthcare ML models that includes:
    - Standard performance metrics
    - Bias and fairness analysis
    - Clinical utility assessment
    - Model interpretability reporting
    """
    
    def __init__(self):
        self.results = {}
        
        # Configure logging and plotting
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        plt.style.use('default')
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_proba):
        """
        Calculate comprehensive performance metrics for healthcare applications
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_proba (array): Predicted probabilities
            
        Returns:
            dict: Comprehensive performance metrics
        """
        self.logger.info("Calculating comprehensive performance metrics...")
        
        metrics = {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_proba),
            'accuracy': np.mean(y_true == y_pred)
        }
        
        # Calculate F2 score (emphasizes recall for healthcare)
        beta = 2
        precision = metrics['precision']
        recall = metrics['recall']
        metrics['f2_score'] = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        
        self.logger.info("Performance metrics calculated successfully")
        return metrics
    
    def generate_confusion_matrix_analysis(self, y_true, y_pred):
        """
        Generate detailed confusion matrix analysis with clinical interpretation
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            
        Returns:
            dict: Confusion matrix analysis results
        """
        self.logger.info("Generating confusion matrix analysis...")
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        analysis = {
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        }
        
        # Clinical impact analysis
        total_patients = len(y_true)
        analysis['clinical_impact'] = {
            'missed_readmissions': fn,
            'unnecessary_interventions': fp,
            'correctly_identified_high_risk': tp,
            'correctly_identified_low_risk': tn
        }
        
        self.logger.info("Confusion matrix analysis completed")
        return analysis
    
    def analyze_model_bias(self, model, X_test, y_test, demographic_data):
        """
        Analyze model performance across different demographic groups
        to detect potential biases
        
        Args:
            model: Trained model
            X_test (DataFrame): Test features
            y_test (array): Test labels
            demographic_data (Series): Demographic group labels
            
        Returns:
            dict: Bias analysis results
        """
        self.logger.info("Analyzing model bias across demographic groups...")
        
        bias_analysis = {}
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        for group in demographic_data.unique():
            group_mask = demographic_data == group
            if group_mask.sum() > 0:  # Ensure group has samples
                group_metrics = self.calculate_comprehensive_metrics(
                    y_test[group_mask], y_pred[group_mask], y_proba[group_mask]
                )
                
                bias_analysis[group] = {
                    'sample_size': group_mask.sum(),
                    'readmission_rate': y_test[group_mask].mean(),
                    'predicted_positive_rate': y_pred[group_mask].mean(),
                    'metrics': group_metrics
                }
        
        # Calculate fairness metrics
        self._calculate_fairness_metrics(bias_analysis)
        
        self.logger.info("Bias analysis completed")
        return bias_analysis
    
    def _calculate_fairness_metrics(self, bias_analysis):
        """
        Calculate statistical fairness metrics across groups
        
        Args:
            bias_analysis (dict): Initial bias analysis results
        """
        groups = list(bias_analysis.keys())
        
        if len(groups) >= 2:
            # Calculate demographic parity differences
            positive_rates = [bias_analysis[g]['predicted_positive_rate'] for g in groups]
            max_positive_rate_diff = max(positive_rates) - min(positive_rates)
            
            # Calculate equal opportunity differences (recall differences)
            recalls = [bias_analysis[g]['metrics']['recall'] for g in groups]
            max_recall_diff = max(recalls) - min(recalls)
            
            fairness_metrics = {
                'demographic_parity_difference': max_positive_rate_diff,
                'equal_opportunity_difference': max_recall_diff,
                'fairness_assessment': 'Fair' if max_positive_rate_diff < 0.1 and max_recall_diff < 0.1 else 'Potential Bias Detected'
            }
            
            # Add fairness metrics to analysis
            for group in groups:
                bias_analysis[group]['fairness_metrics'] = fairness_metrics
    
    def generate_roc_pr_curves(self, y_true, y_proba, save_path=None):
        """
        Generate ROC and Precision-Recall curves for model evaluation
        
        Args:
            y_true (array): True labels
            y_proba (array): Predicted probabilities
            save_path (str): Path to save the plots
            
        Returns:
            dict: Curve data and AUC values
        """
        self.logger.info("Generating ROC and Precision-Recall curves...")
        
        # ROC Curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        # Precision-Recall Curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc = np.trapz(precision, recall)
        
        curve_data = {
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds, 'auc': roc_auc},
            'pr_curve': {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds, 'auc': pr_auc}
        }
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curve
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True)
        
        # Precision-Recall Curve
        ax2.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        baseline = y_true.mean()
        ax2.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline (Precision = {baseline:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Curves saved to {save_path}")
        
        plt.show()
        
        return curve_data
    
    def create_clinical_utility_report(self, confusion_analysis, metrics):
        """
        Generate a clinical utility report explaining the model's
        practical impact in healthcare settings
        
        Args:
            confusion_analysis (dict): Confusion matrix analysis
            metrics (dict): Performance metrics
            
        Returns:
            dict: Clinical utility assessment
        """
        self.logger.info("Generating clinical utility report...")
        
        clinical_impact = confusion_analysis['clinical_impact']
        
        utility_report = {
            'summary': {
                'model_performance': 'Clinically Acceptable' if metrics['f2_score'] > 0.7 else 'Needs Improvement',
                'key_strength': 'High Recall - Minimizes missed readmissions' if metrics['recall'] > 0.7 else 'Needs higher recall',
                'main_concern': 'False positives may lead to unnecessary interventions' if confusion_analysis['false_positive_rate'] > 0.3 else 'Acceptable false positive rate'
            },
            'recommendations': [
                'Use for initial patient screening with clinical review',
                'Focus interventions on high-probability predictions',
                'Monitor real-world impact on readmission rates'
            ],
            'implementation_guidance': {
                'threshold_adjustment': 'Consider lowering threshold if recall is prioritized',
                'clinical_workflow': 'Integrate with discharge planning process',
                'monitoring': 'Track intervention effectiveness monthly'
            }
        }
        
        return utility_report
    
    def generate_comprehensive_report(self, model, X_test, y_test, demographic_data=None, 
                                   save_path='evaluation/model_evaluation_report.txt'):
        """
        Generate a comprehensive evaluation report covering all aspects
        of model performance and clinical utility
        
        Args:
            model: Trained model
            X_test (DataFrame): Test features
            y_test (array): Test labels
            demographic_data (Series): Demographic data for bias analysis
            save_path (str): Path to save the report
        """
        self.logger.info("Generating comprehensive evaluation report...")
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate all metrics
        metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_proba)
        confusion_analysis = self.generate_confusion_matrix_analysis(y_test, y_pred)
        
        # Generate curves
        curve_data = self.generate_roc_pr_curves(y_test, y_proba)
        
        # Bias analysis if demographic data provided
        bias_analysis = None
        if demographic_data is not None:
            bias_analysis = self.analyze_model_bias(model, X_test, y_test, demographic_data)
        
        # Clinical utility report
        clinical_report = self.create_clinical_utility_report(confusion_analysis, metrics)
        
        # Compile comprehensive results
        comprehensive_results = {
            'performance_metrics': metrics,
            'confusion_analysis': confusion_analysis,
            'bias_analysis': bias_analysis,
            'clinical_utility': clinical_report,
            'curve_data': curve_data
        }
        
        # Save detailed report
        self._save_evaluation_report(comprehensive_results, save_path)
        
        self.logger.info(f"Comprehensive evaluation report saved to {save_path}")
        return comprehensive_results
    
    def _save_evaluation_report(self, results, filepath):
        """
        Save a detailed text report of the evaluation results
        
        Args:
            results (dict): Comprehensive evaluation results
            filepath (str): Path to save the report
        """
        with open(filepath, 'w') as f:
            f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Performance Metrics Section
            f.write("1. PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            metrics = results['performance_metrics']
            for metric, value in metrics.items():
                f.write(f"{metric.upper()}: {value:.4f}\n")
            
            # Confusion Matrix Section
            f.write("\n2. CONFUSION MATRIX ANALYSIS\n")
            f.write("-" * 20 + "\n")
            cm_analysis = results['confusion_analysis']
            f.write(f"True Positives: {cm_analysis['true_positives']}\n")
            f.write(f"False Negatives: {cm_analysis['false_negatives']}\n")
            f.write(f"False Positives: {cm_analysis['false_positives']}\n")
            f.write(f"True Negatives: {cm_analysis['true_negatives']}\n")
            f.write(f"Sensitivity: {cm_analysis['sensitivity']:.4f}\n")
            f.write(f"Specificity: {cm_analysis['specificity']:.4f}\n")
            
            # Bias Analysis Section
            if results['bias_analysis']:
                f.write("\n3. BIAS AND FAIRNESS ANALYSIS\n")
                f.write("-" * 20 + "\n")
                for group, analysis in results['bias_analysis'].items():
                    f.write(f"\nGroup {group}:\n")
                    f.write(f"  Sample Size: {analysis['sample_size']}\n")
                    f.write(f"  Readmission Rate: {analysis['readmission_rate']:.4f}\n")
                    f.write(f"  F1-Score: {analysis['metrics']['f1']:.4f}\n")
                    if 'fairness_metrics' in analysis:
                        f.write(f"  Fairness: {analysis['fairness_metrics']['fairness_assessment']}\n")
            
            # Clinical Utility Section
            f.write("\n4. CLINICAL UTILITY ASSESSMENT\n")
            f.write("-" * 20 + "\n")
            clinical = results['clinical_utility']
            f.write(f"Overall Assessment: {clinical['summary']['model_performance']}\n")
            f.write(f"Key Strength: {clinical['summary']['key_strength']}\n")
            f.write(f"Main Concern: {clinical['summary']['main_concern']}\n")
            
            f.write("\nRecommendations:\n")
            for rec in clinical['recommendations']:
                f.write(f"  - {rec}\n")
            
            f.write("\nImplementation Guidance:\n")
            for key, guidance in clinical['implementation_guidance'].items():
                f.write(f"  {key.replace('_', ' ').title()}: {guidance}\n")

# Example usage
if __name__ == "__main__":
    # Simulate test data
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    y_proba = np.random.random(n_samples)
    y_pred = (y_proba > 0.5).astype(int)
    
    # Simulate demographic data
    demographic_data = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.6, 0.3, 0.1])
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Generate comprehensive report
    results = evaluator.generate_comprehensive_report(
        model=None,  # In practice, this would be a trained model
        X_test=pd.DataFrame(np.random.random((n_samples, 5))),  # Simulated features
        y_test=y_true,
        demographic_data=pd.Series(demographic_data),
        save_path='evaluation/comprehensive_report.txt'
    )
    
    print("Evaluation completed!")
    print(f"F1-Score: {results['performance_metrics']['f1']:.4f}")
    print(f"F2-Score: {results['performance_metrics']['f2_score']:.4f}")
    print(f"AUC-ROC: {results['performance_metrics']['auc_roc']:.4f}")