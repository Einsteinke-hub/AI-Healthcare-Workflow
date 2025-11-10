import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

class HealthcareDataPreprocessor:
    """
    A comprehensive preprocessor for healthcare data that handles:
    - Missing value imputation
    - Feature engineering for clinical variables
    - Normalization and encoding
    - Bias detection and mitigation
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        
        # Configure logging for data processing tracking
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_data_from_sources(self):
        """
        Simulate data extraction from multiple hospital sources
        In real implementation, this would connect to EHR databases
        
        Returns:
            pandas.DataFrame: Combined dataset from all sources
        """
        self.logger.info("Loading data from EHR systems...")
        
        # Simulate EHR data extraction
        clinical_data = {
            'patient_id': range(1000),
            'age': np.random.randint(18, 90, 1000),
            'length_of_stay': np.random.randint(1, 30, 1000),
            'num_medications': np.random.randint(1, 15, 1000),
            'num_lab_procedures': np.random.randint(1, 25, 1000),
            'number_diagnoses': np.random.randint(1, 10, 1000),
            'readmitted': np.random.choice([0, 1], 1000, p=[0.85, 0.15])
        }
        
        df = pd.DataFrame(clinical_data)
        self.logger.info(f"Loaded dataset with {len(df)} patients")
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values using appropriate strategies for different data types
        
        Args:
            df (pandas.DataFrame): Raw input data
            
        Returns:
            pandas.DataFrame: Data with missing values handled
        """
        self.logger.info("Handling missing values...")
        
        # Simulate missing data for demonstration
        mask = np.random.random(len(df)) < 0.05  # 5% missing
        df.loc[mask, 'num_medications'] = np.nan
        
        # Numerical features: median imputation
        num_features = ['age', 'length_of_stay', 'num_medications', 
                       'num_lab_procedures', 'number_diagnoses']
        
        df[num_features] = self.imputer_num.fit_transform(df[num_features])
        
        self.logger.info("Missing values imputed successfully")
        return df
    
    def engineer_clinical_features(self, df):
        """
        Create clinically relevant features for readmission prediction
        
        Args:
            df (pandas.DataFrame): Cleaned input data
            
        Returns:
            pandas.DataFrame: Data with engineered features
        """
        self.logger.info("Engineering clinical features...")
        
        # Calculate Charlson Comorbidity Index (simplified)
        df['comorbidity_index'] = np.where(df['number_diagnoses'] > 5, 2,
                                         np.where(df['number_diagnoses'] > 2, 1, 0))
        
        # Medication complexity score
        df['medication_complexity'] = np.where(df['num_medications'] > 10, 'high',
                                             np.where(df['num_medications'] > 5, 'medium', 'low'))
        
        # Length of stay categories
        df['stay_category'] = np.where(df['length_of_stay'] > 14, 'prolonged',
                                     np.where(df['length_of_stay'] > 7, 'medium', 'short'))
        
        # Age groups for clinical relevance
        df['age_group'] = pd.cut(df['age'], 
                               bins=[0, 30, 50, 65, 90],
                               labels=['young', 'adult', 'senior', 'elderly'])
        
        self.logger.info("Clinical features engineered successfully")
        return df
    
    def encode_categorical_features(self, df):
        """
        Encode categorical variables for machine learning
        
        Args:
            df (pandas.DataFrame): Data with categorical features
            
        Returns:
            pandas.DataFrame: Data with encoded categorical features
        """
        self.logger.info("Encoding categorical features...")
        
        categorical_features = ['medication_complexity', 'stay_category', 'age_group']
        
        for feature in categorical_features:
            if feature in df.columns:
                self.label_encoders[feature] = LabelEncoder()
                df[feature] = self.label_encoders[feature].fit_transform(df[feature])
        
        self.logger.info("Categorical features encoded successfully")
        return df
    
    def normalize_numerical_features(self, df):
        """
        Normalize numerical features to standard scale
        
        Args:
            df (pandas.DataFrame): Data with numerical features
            
        Returns:
            pandas.DataFrame: Data with normalized numerical features
        """
        self.logger.info("Normalizing numerical features...")
        
        numerical_features = ['age', 'length_of_stay', 'num_medications', 
                            'num_lab_procedures', 'number_diagnoses']
        
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        self.logger.info("Numerical features normalized successfully")
        return df
    
    def detect_data_bias(self, df):
        """
        Analyze dataset for potential biases across demographic groups
        
        Args:
            df (pandas.DataFrame): Processed dataset
            
        Returns:
            dict: Bias analysis results
        """
        self.logger.info("Analyzing dataset for biases...")
        
        # Simulate demographic data for bias analysis
        np.random.seed(42)
        df['demographic_group'] = np.random.choice(['A', 'B', 'C'], len(df), p=[0.6, 0.3, 0.1])
        
        bias_analysis = {}
        
        # Analyze readmission rates by demographic group
        for group in df['demographic_group'].unique():
            group_data = df[df['demographic_group'] == group]
            readmission_rate = group_data['readmitted'].mean()
            bias_analysis[group] = {
                'count': len(group_data),
                'readmission_rate': readmission_rate,
                'representation_pct': len(group_data) / len(df) * 100
            }
        
        self.logger.info("Bias analysis completed")
        return bias_analysis
    
    def run_complete_pipeline(self):
        """
        Execute the complete data preprocessing pipeline
        
        Returns:
            tuple: (processed_features, target, bias_report)
        """
        self.logger.info("Starting complete data preprocessing pipeline...")
        
        # Step 1: Load data
        df = self.load_data_from_sources()
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Feature engineering
        df = self.engineer_clinical_features(df)
        
        # Step 4: Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Step 5: Normalize numerical features
        df = self.normalize_numerical_features(df)
        
        # Step 6: Bias detection
        bias_report = self.detect_data_bias(df)
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col not in 
                         ['patient_id', 'readmitted', 'demographic_group']]
        
        X = df[feature_columns]
        y = df['readmitted']
        
        self.logger.info("Data preprocessing pipeline completed successfully")
        
        return X, y, bias_report

# Example usage
if __name__ == "__main__":
    preprocessor = HealthcareDataPreprocessor()
    X_processed, y_processed, bias_report = preprocessor.run_complete_pipeline()
    
    print("Preprocessing completed!")
    print(f"Processed features shape: {X_processed.shape}")
    print(f"Target distribution: {y_processed.value_counts().to_dict()}")
    print("\nBias Analysis Report:")
    for group, stats in bias_report.items():
        print(f"Group {group}: {stats}")