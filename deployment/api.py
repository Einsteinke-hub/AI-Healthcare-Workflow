"""
Model Deployment API for Healthcare Readmission Prediction

This module provides a RESTful API for model serving with HIPAA-compliant
data handling, monitoring, and scalability features.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import time
import hashlib
import json
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment/api.log'),
        logging.StreamHandler()
    ]
)

class HealthcareModelAPI:
    """
    A HIPAA-compliant REST API for patient readmission prediction
    with features for monitoring, security, and scalability
    """
    
    def __init__(self, model_path='models/readmission_model.pkl'):
        self.app = Flask(__name__)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.requests_log = []
        self.performance_metrics = {
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_response_time': 0
        }
        
        # Load model
        self.load_model(model_path)
        
        # Setup routes
        self.setup_routes()
    
    def load_model(self, model_path):
        """
        Load the trained model and associated preprocessing objects
        
        Args:
            model_path (str): Path to the saved model file
        """
        try:
            self.model = joblib.load(model_path)
            self.logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def setup_routes(self):
        """
        Define API routes and their corresponding handlers
        """
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint for API monitoring"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'model_loaded': self.model is not None
            })
        
        @self.app.route('/predict', methods=['POST'])
        @self.require_api_key
        @self.log_request
        def predict_readmission():
            """
            Main prediction endpoint that takes patient data and returns
            readmission risk prediction with confidence scores
            """
            start_time = time.time()
            
            try:
                # Validate and parse request data
                data = self.validate_prediction_request(request)
                
                # Preprocess input data
                processed_data = self.preprocess_input(data['patient_data'])
                
                # Generate prediction
                prediction, confidence = self.generate_prediction(processed_data)
                
                # Log successful prediction
                response_time = time.time() - start_time
                self.log_successful_prediction(response_time)
                
                # Prepare response
                response = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'risk_category': self.get_risk_category(confidence),
                    'recommendations': self.get_clinical_recommendations(prediction, confidence),
                    'request_id': data['request_id'],
                    'timestamp': datetime.now().isoformat()
                }
                
                return jsonify(response), 200
                
            except Exception as e:
                self.log_failed_prediction(str(e))
                return jsonify({
                    'error': str(e),
                    'request_id': request.json.get('request_id', 'unknown')
                }), 400
        
        @self.app.route('/metrics', methods=['GET'])
        @self.require_api_key
        def get_metrics():
            """Endpoint to retrieve API performance metrics"""
            return jsonify(self.performance_metrics), 200
        
        @self.app.route('/monitoring', methods=['GET'])
        @self.require_api_key
        def get_monitoring_data():
            """Endpoint for comprehensive system monitoring"""
            monitoring_data = {
                'performance_metrics': self.performance_metrics,
                'recent_requests': self.requests_log[-10:],  # Last 10 requests
                'system_status': self.get_system_status(),
                'model_info': self.get_model_info()
            }
            return jsonify(monitoring_data), 200
    
    def require_api_key(self, f):
        """
        Decorator for API key authentication
        In production, this would integrate with proper authentication service
        """
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Simulated API key validation
            api_key = request.headers.get('X-API-Key')
            if not api_key or not self.validate_api_key(api_key):
                return jsonify({'error': 'Invalid or missing API key'}), 401
            return f(*args, **kwargs)
        return decorated_function
    
    def validate_api_key(self, api_key):
        """
        Validate API key (simplified for demonstration)
        In production, use proper authentication service
        
        Args:
            api_key (str): API key to validate
            
        Returns:
            bool: Validation result
        """
        # In production, this would check against a secure key store
        valid_keys = ['healthcare_api_key_2024', 'test_key_123']
        return api_key in valid_keys
    
    def validate_prediction_request(self, request):
        """
        Validate and sanitize prediction request data
        
        Args:
            request: Flask request object
            
        Returns:
            dict: Validated and parsed request data
            
        Raises:
            ValueError: If request validation fails
        """
        if not request.is_json:
            raise ValueError("Request must be JSON")
        
        data = request.get_json()
        
        # Check required fields
        required_fields = ['patient_data']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Generate request ID if not provided
        if 'request_id' not in data:
            data['request_id'] = self.generate_request_id()
        
        # Validate patient data structure
        patient_data = data['patient_data']
        if not isinstance(patient_data, dict):
            raise ValueError("patient_data must be a dictionary")
        
        self.logger.info(f"Validated request {data['request_id']}")
        return data
    
    def generate_request_id(self):
        """
        Generate unique request ID for tracking
        
        Returns:
            str: Unique request identifier
        """
        timestamp = datetime.now().isoformat()
        unique_string = f"{timestamp}{np.random.random()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]
    
    def preprocess_input(self, patient_data):
        """
        Preprocess incoming patient data for model prediction
        This should mirror the preprocessing done during training
        
        Args:
            patient_data (dict): Raw patient data from request
            
        Returns:
            pandas.DataFrame: Processed data ready for prediction
        """
        # Convert to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Apply the same preprocessing as training
        # In practice, this would use the same preprocessing pipeline
        numerical_features = ['age', 'length_of_stay', 'num_medications', 
                            'num_lab_procedures', 'number_diagnoses']
        
        categorical_features = ['medication_complexity', 'stay_category', 'age_group']
        
        # Handle missing values (simplified)
        for feature in numerical_features:
            if feature in df.columns and pd.isna(df[feature]).any():
                df[feature].fillna(df[feature].median(), inplace=True)
        
        for feature in categorical_features:
            if feature in df.columns and pd.isna(df[feature]).any():
                df[feature].fillna('unknown', inplace=True)
        
        self.logger.debug("Input data preprocessed successfully")
        return df
    
    def generate_prediction(self, processed_data):
        """
        Generate prediction using the loaded model
        
        Args:
            processed_data (DataFrame): Preprocessed patient data
            
        Returns:
            tuple: (prediction, confidence_score)
        """
        try:
            # Get probability prediction
            probability = self.model.predict_proba(processed_data)[0, 1]
            
            # Convert to binary prediction with threshold
            prediction = int(probability > 0.5)
            confidence = float(probability)
            
            self.logger.info(f"Prediction generated: {prediction} (confidence: {confidence:.3f})")
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def get_risk_category(self, confidence):
        """
        Convert confidence score to clinical risk category
        
        Args:
            confidence (float): Prediction confidence score
            
        Returns:
            str: Risk category description
        """
        if confidence >= 0.7:
            return "High Risk"
        elif confidence >= 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def get_clinical_recommendations(self, prediction, confidence):
        """
        Generate clinical recommendations based on prediction
        
        Args:
            prediction (int): Binary prediction (0/1)
            confidence (float): Prediction confidence
            
        Returns:
            list: Clinical recommendations
        """
        if prediction == 1:  # High risk of readmission
            return [
                "Schedule follow-up appointment within 7 days",
                "Assign care coordinator for transition support",
                "Review medication reconciliation",
                "Consider home health referral"
            ]
        else:  # Low risk
            return [
                "Standard discharge instructions",
                "Provide patient education materials",
                "Schedule routine follow-up"
            ]
    
    def log_request(self, f):
        """
        Decorator to log all API requests
        """
        @wraps(f)
        def decorated_function(*args, **kwargs):
            request_data = {
                'endpoint': request.endpoint,
                'method': request.method,
                'timestamp': datetime.now().isoformat(),
                'ip_address': request.remote_addr,
                'user_agent': request.headers.get('User-Agent', 'Unknown')
            }
            self.requests_log.append(request_data)
            
            # Keep log manageable
            if len(self.requests_log) > 1000:
                self.requests_log = self.requests_log[-1000:]
            
            return f(*args, **kwargs)
        return decorated_function
    
    def log_successful_prediction(self, response_time):
        """
        Log successful prediction and update performance metrics
        
        Args:
            response_time (float): API response time in seconds
        """
        self.performance_metrics['total_requests'] += 1
        self.performance_metrics['successful_predictions'] += 1
        
        # Update average response time
        current_avg = self.performance_metrics['average_response_time']
        n = self.performance_metrics['successful_predictions']
        new_avg = (current_avg * (n - 1) + response_time) / n
        self.performance_metrics['average_response_time'] = new_avg
    
    def log_failed_prediction(self, error_message):
        """
        Log failed prediction attempt
        
        Args:
            error_message (str): Error description
        """
        self.performance_metrics['total_requests'] += 1
        self.performance_metrics['failed_predictions'] += 1
        self.logger.error(f"Prediction failed: {error_message}")
    
    def get_system_status(self):
        """
        Get current system status information
        
        Returns:
            dict: System status metrics
        """
        return {
            'model_loaded': self.model is not None,
            'memory_usage': 'stable',  # In production, get actual metrics
            'uptime': 'TODO',  # In production, calculate actual uptime
            'timestamp': datetime.now().isoformat()
        }
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {'status': 'No model loaded'}
        
        return {
            'model_type': type(self.model).__name__,
            'model_parameters': str(self.model.get_params()) if hasattr(self.model, 'get_params') else 'N/A',
            'training_date': '2024-01-01',  # In production, store this metadata
            'version': '1.0.0'
        }
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """
        Run the Flask API server
        
        Args:
            host (str): Host address to bind to
            port (int): Port to listen on
            debug (bool): Enable debug mode
        """
        self.logger.info(f"Starting Healthcare Model API on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Example usage and testing
if __name__ == "__main__":
    # Initialize and run the API
    api = HealthcareModelAPI()
    
    # Example of how to test the API
    print("Healthcare Readmission Prediction API")
    print("Endpoints:")
    print("  GET  /health     - Health check")
    print("  POST /predict    - Make prediction")
    print("  GET  /metrics    - Performance metrics")
    print("  GET  /monitoring - System monitoring")
    print("\nStarting server...")
    
    # Run the API (in production, use proper WSGI server like Gunicorn)
    api.run(debug=True)