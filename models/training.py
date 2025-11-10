"""
Model Training Module for Readmission Prediction

This module handles model selection, training, hyperparameter tuning, and validation
using stratified cross-validation to address class imbalance.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

class ReadmissionPredictor:
    """
    A machine learning model trainer for patient readmission prediction
    that includes multiple algorithms with hyperparameter optimization
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_params = {}
        self.cv_results = {}
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_models(self):
        """
        Initialize multiple candidate models with their parameter grids
        for comprehensive comparison
        """
        self.logger.info("Initializing candidate models...")
        
        self.models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, None],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 4]
                }
            }
        }
        
        self.logger.info(f"Initialized {len(self.models)} candidate models")
    
    def perform_stratified_split(self, X, y, test_size=0.2, val_size=0.15):
        """
        Create stratified train/validation/test splits to maintain
        class distribution across all sets
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector
            test_size (float): Proportion for test set
            val_size (float): Proportion for validation set
            
        Returns:
            dict: Split indices for train, validation, and test sets
        """
        self.logger.info("Performing stratified data splitting...")
        
        # Calculate split sizes
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_test - n_val
        
        # Create stratified splits
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Get one split for train/val/test
        for train_val_idx, test_idx in skf.split(X, y):
            # Further split train_val into train and validation
            X_train_val, y_train_val = X.iloc[train_val_idx], y.iloc[train_val_idx]
            
            skf_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            for train_idx, val_idx in skf_inner.split(X_train_val, y_train_val):
                splits = {
                    'X_train': X.iloc[train_val_idx[train_idx]],
                    'X_val': X.iloc[train_val_idx[val_idx]],
                    'X_test': X.iloc[test_idx],
                    'y_train': y.iloc[train_val_idx[train_idx]],
                    'y_val': y.iloc[train_val_idx[val_idx]],
                    'y_test': y.iloc[test_idx]
                }
                break
            break
        
        self.logger.info(f"Train set: {len(splits['y_train'])} samples")
        self.logger.info(f"Validation set: {len(splits['y_val'])} samples") 
        self.logger.info(f"Test set: {len(splits['y_test'])} samples")
        
        return splits
    
    def tune_hyperparameters(self, model_name, model_config, X_train, y_train):
        """
        Perform hyperparameter tuning using GridSearchCV with
        stratified cross-validation
        
        Args:
            model_name (str): Name of the model
            model_config (dict): Model configuration including parameters
            X_train (array-like): Training features
            y_train (array-like): Training targets
            
        Returns:
            tuple: (best_estimator, best_score, best_params)
        """
        self.logger.info(f"Tuning hyperparameters for {model_name}...")
        
        # Use stratified CV for hyperparameter tuning
        cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            estimator=model_config['model'],
            param_grid=model_config['params'],
            cv=cv_stratified,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        self.logger.info(f"Best {model_name} params: {grid_search.best_params_}")
        self.logger.info(f"Best {model_name} CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_
    
    def evaluate_model_interpretability(self, model, model_name, feature_names):
        """
        Evaluate model interpretability for clinical adoption
        
        Args:
            model: Trained model instance
            model_name (str): Name of the model
            feature_names (list): List of feature names
            
        Returns:
            dict: Interpretability metrics
        """
        self.logger.info(f"Evaluating interpretability for {model_name}...")
        
        interpretability_scores = {}
        
        if model_name == 'logistic_regression':
            # Logistic regression provides coefficient interpretability
            if hasattr(model, 'coef_'):
                feature_importance = abs(model.coef_[0])
                interpretability_scores = {
                    'score': 9,  # High interpretability
                    'reason': 'Linear model with clear coefficient interpretation',
                    'top_features': list(zip(feature_names, feature_importance))
                }
        
        elif model_name == 'random_forest':
            # Random forests provide feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                interpretability_scores = {
                    'score': 7,  # Medium interpretability
                    'reason': 'Feature importance available but complex interactions',
                    'top_features': list(zip(feature_names, feature_importance))
                }
        
        else:
            # Gradient boosting is less interpretable
            interpretability_scores = {
                'score': 5,  # Lower interpretability
                'reason': 'Complex ensemble method with limited direct interpretability',
                'top_features': []
            }
        
        return interpretability_scores
    
    def train_final_model(self, splits, feature_names):
        """
        Train and evaluate multiple models, then select the best one
        based on performance and interpretability
        
        Args:
            splits (dict): Train/validation/test splits
            feature_names (list): Names of features for interpretability
            
        Returns:
            tuple: (best_model, evaluation_results, interpretability_scores)
        """
        self.logger.info("Starting model training and selection...")
        
        self.initialize_models()
        evaluation_results = {}
        interpretability_scores = {}
        
        # Train and evaluate each model
        for model_name, model_config in self.models.items():
            self.logger.info(f"Training {model_name}...")
            
            # Hyperparameter tuning
            best_estimator, best_score, best_params = self.tune_hyperparameters(
                model_name, model_config, splits['X_train'], splits['y_train']
            )
            
            # Validation set evaluation
            y_val_pred = best_estimator.predict(splits['X_val'])
            y_val_proba = best_estimator.predict_proba(splits['X_val'])[:, 1]
            
            val_metrics = {
                'precision': precision_score(splits['y_val'], y_val_pred),
                'recall': recall_score(splits['y_val'], y_val_pred),
                'f1': f1_score(splits['y_val'], y_val_pred),
                'auc_roc': roc_auc_score(splits['y_val'], y_val_proba),
                'cv_score': best_score,
                'best_params': best_params
            }
            
            # Interpretability evaluation
            interpretability = self.evaluate_model_interpretability(
                best_estimator, model_name, feature_names
            )
            
            evaluation_results[model_name] = val_metrics
            interpretability_scores[model_name] = interpretability
            
            self.logger.info(f"{model_name} - F1: {val_metrics['f1']:.4f}, "
                           f"AUC-ROC: {val_metrics['auc_roc']:.4f}")
        
        # Select best model (prioritizing interpretability in healthcare)
        best_model_name = self.select_best_model(evaluation_results, interpretability_scores)
        self.best_model = self.models[best_model_name]['model'].set_params(
            **evaluation_results[best_model_name]['best_params']
        )
        self.best_model.fit(
            pd.concat([splits['X_train'], splits['X_val']]),
            pd.concat([splits['y_train'], splits['y_val']])
        )
        
        self.logger.info(f"Selected best model: {best_model_name}")
        
        return self.best_model, evaluation_results, interpretability_scores
    
    def select_best_model(self, evaluation_results, interpretability_scores):
        """
        Select the best model considering both performance and interpretability
        for healthcare applications
        
        Args:
            evaluation_results (dict): Model performance metrics
            interpretability_scores (dict): Model interpretability scores
            
        Returns:
            str: Name of the selected best model
        """
        self.logger.info("Selecting best model based on performance and interpretability...")
        
        # Calculate composite score (70% performance, 30% interpretability)
        model_scores = {}
        
        for model_name in evaluation_results.keys():
            # Normalize F1 score (performance)
            f1_scores = [results['f1'] for results in evaluation_results.values()]
            normalized_f1 = (evaluation_results[model_name]['f1'] - min(f1_scores)) / (max(f1_scores) - min(f1_scores))
            
            # Normalize interpretability score
            interpret_scores = [scores['score'] for scores in interpretability_scores.values()]
            normalized_interpret = (interpretability_scores[model_name]['score'] - min(interpret_scores)) / (max(interpret_scores) - min(interpret_scores))
            
            # Composite score
            composite_score = 0.7 * normalized_f1 + 0.3 * normalized_interpret
            model_scores[model_name] = composite_score
        
        best_model = max(model_scores, key=model_scores.get)
        
        self.logger.info("Model selection scores:")
        for model, score in model_scores.items():
            self.logger.info(f"  {model}: {score:.4f}")
        
        return best_model
    
    def save_model(self, filepath='models/readmission_model.pkl'):
        """
        Save the trained model for deployment
        
        Args:
            filepath (str): Path to save the model
        """
        if self.best_model is not None:
            joblib.dump(self.best_model, filepath)
            self.logger.info(f"Model saved to {filepath}")
        else:
            self.logger.warning("No trained model to save")

# Example usage
if __name__ == "__main__":
    # Simulate data loading (in practice, this would come from preprocessing)
    from data.preprocessing import HealthcareDataPreprocessor
    
    # Initialize and run preprocessing
    preprocessor = HealthcareDataPreprocessor()
    X, y, bias_report = preprocessor.run_complete_pipeline()
    
    # Initialize and run model training
    trainer = ReadmissionPredictor()
    splits = trainer.perform_stratified_split(X, y)
    
    # Train models and select best one
    best_model, results, interpretability = trainer.train_final_model(
        splits, feature_names=X.columns.tolist()
    )
    
    # Save the best model
    trainer.save_model()
    
    print("\n=== MODEL TRAINING COMPLETED ===")
    print("Final Model Performance Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  Interpretability: {interpretability[model_name]['score']}/10")