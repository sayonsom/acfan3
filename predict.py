import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, concatenate
import joblib
import os
import logging
from typing import Tuple, Optional, Union, List, Dict
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'predictions_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

class CustomFeatureLayer(Layer):
    """Custom layer for feature engineering"""
    def __init__(self, **kwargs):
        super(CustomFeatureLayer, self).__init__(**kwargs)

    def call(self, inputs):
        interaction = inputs[:, 0] * inputs[:, 1]
        interaction = tf.expand_dims(interaction, -1)
        squares = tf.square(inputs)
        return concatenate([inputs, interaction, squares])

    def get_config(self):
        return super(CustomFeatureLayer, self).get_config()

class ComfortBasedPredictor:
    """AC Fan predictor with comfort temperature preference"""
    
    def __init__(
        self,
        model_path: str,
        scaler_x_path: str,
        scaler_y_path: str,
        temp_bounds: Tuple[float, float] = (18.0, 30.0),
        velocity_bounds: Tuple[float, float] = (0.1, 1.2)
    ):
        self.model_path = model_path
        self.scaler_x_path = scaler_x_path
        self.scaler_y_path = scaler_y_path
        self.temp_bounds = temp_bounds
        self.velocity_bounds = velocity_bounds
        
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        
        # Updated input validation bounds
        self.input_bounds = {
            'outdoor_temp': (-10.0, 50.0),     # Outdoor temperature range
            'comfort_temp': (16.0, 32.0),      # Updated comfort temperature range based on lookup table
            'pmv': (-3.0, 3.0)                 # PMV range
        }
        
        # PMV calculation constants based on lookup table
        self.PMV_REFERENCE_TEMP = 24.0         # Reference temperature for PMV calculation
        self.PMV_RATE = 0.3                    # PMV change per degree Celsius (from lookup table)
        
    def calculate_pmv(self, comfort_temp: float) -> float:
        """
        Calculate PMV based on comfort temperature
        PMV changes by 0.3 for every degree difference from reference temperature (24°C)
        Based on empirical lookup table values
        
        Args:
            comfort_temp: Desired comfort temperature in Celsius
            
        Returns:
            float: Calculated PMV value
        """
        # Calculate temperature difference from reference (24°C)
        temp_difference = comfort_temp - self.PMV_REFERENCE_TEMP
        
        # Calculate PMV (0.3 per degree difference)
        pmv = temp_difference * self.PMV_RATE
        
        # Clip PMV to valid range
        return np.clip(pmv, -3.0, 3.0)
    
    def get_pmv_description(self, pmv: float) -> str:
        """
        Get thermal sensation description based on PMV value
        """
        if pmv <= -2.5:
            return "Cold"
        elif pmv <= -1.5:
            return "Cool"
        elif pmv <= -0.5:
            return "Slightly Cool"
        elif pmv < 0.5:
            return "Neutral"
        elif pmv < 1.5:
            return "Slightly Warm"
        elif pmv < 2.5:
            return "Warm"
        else:
            return "Hot"
    
    def load(self) -> bool:
        """Load the model and scalers"""
        try:
            custom_objects = {
                'CustomFeatureLayer': CustomFeatureLayer
            }
            
            logging.info("Loading model and scalers...")
            self.model = load_model(self.model_path, custom_objects=custom_objects)
            self.scaler_x = joblib.load(self.scaler_x_path)
            self.scaler_y = joblib.load(self.scaler_y_path)
            logging.info("Model and scalers loaded successfully!")
            return True
            
        except Exception as e:
            logging.error(f"Error loading model or scalers: {e}")
            return False
    
    def validate_input(
        self,
        outdoor_temp: float,
        comfort_temp: float
    ) -> Tuple[bool, str]:
        """Validate input values"""
        if not self.input_bounds['outdoor_temp'][0] <= outdoor_temp <= self.input_bounds['outdoor_temp'][1]:
            return False, f"Outdoor temperature {outdoor_temp}°C is outside valid range {self.input_bounds['outdoor_temp']}"
            
        if not self.input_bounds['comfort_temp'][0] <= comfort_temp <= self.input_bounds['comfort_temp'][1]:
            return False, f"Comfort temperature {comfort_temp}°C is outside valid range {self.input_bounds['comfort_temp']}"
            
        return True, ""
    
    def predict_with_comfort(
        self,
        outdoor_temp: float,
        comfort_temp: float = 24.0,
        validate: bool = True
    ) -> Dict[str, Union[float, str, bool]]:
        """
        Make prediction based on outdoor temperature and desired comfort temperature
        
        Args:
            outdoor_temp: Outdoor temperature in Celsius
            comfort_temp: Desired comfort temperature in Celsius (default: 24.0)
            validate: Whether to validate inputs
            
        Returns:
            Dictionary containing predictions and metadata
        """
        result = {
            'success': False,
            'temperature': None,
            'velocity': None,
            'pmv': None,
            'thermal_sensation': None,
            'message': '',
            'input_valid': True,
            'comfort_temp': comfort_temp
        }
        
        try:
            # Load model if needed
            if self.model is None:
                if not self.load():
                    result['message'] = "Failed to load model and scalers"
                    return result
            
            # Validate input if requested
            if validate:
                is_valid, error_msg = self.validate_input(outdoor_temp, comfort_temp)
                if not is_valid:
                    result['input_valid'] = False
                    result['message'] = error_msg
                    return result
            
            # Calculate PMV based on comfort temperature
            pmv = self.calculate_pmv(comfort_temp)
            
            # Prepare input
            input_data = np.array([[outdoor_temp, pmv]])
            input_scaled = self.scaler_x.transform(input_data)
            
            # Make prediction
            predictions_scaled = self.model.predict(input_scaled, verbose=0)
            
            # Process predictions
            if isinstance(predictions_scaled, list):
                predictions = self.scaler_y.inverse_transform(
                    np.column_stack((predictions_scaled[0], predictions_scaled[1]))
                )
            else:
                predictions = self.scaler_y.inverse_transform(predictions_scaled)
            
            # Get thermal sensation description
            thermal_sensation = self.get_pmv_description(pmv)
            
            result.update({
                'success': True,
                'temperature': float(predictions[0, 0]),
                'velocity': float(predictions[0, 1]),
                'pmv': float(pmv),
                'thermal_sensation': thermal_sensation,
                'message': 'Prediction successful'
            })
            
            return result
            
        except Exception as e:
            result['message'] = f"Error during prediction: {str(e)}"
            return result

def main():
    """Example usage of the ComfortBasedPredictor"""
    
    # Define paths
    MODEL_DIR = 'models'
    SCALER_DIR = 'scalers'
    
    model_path = os.path.join(MODEL_DIR, 'final_model.keras')
    scaler_x_path = os.path.join(SCALER_DIR, 'scaler_X.joblib')
    scaler_y_path = os.path.join(SCALER_DIR, 'scaler_y.joblib')
    
    # Create predictor instance
    predictor = ComfortBasedPredictor(
        model_path=model_path,
        scaler_x_path=scaler_x_path,
        scaler_y_path=scaler_y_path
    )
    
    # Example predictions with different comfort temperatures
    test_cases = [
        {'outdoor_temp': 32.0, 'comfort_temp': 24.0},  # Reference comfort (PMV = 0)
        {'outdoor_temp': 35.0, 'comfort_temp': 20.0},  # Cool preference
        {'outdoor_temp': 30.0, 'comfort_temp': 28.0},  # Warm preference
        {'outdoor_temp': 33.0, 'comfort_temp': 16.0},  # Minimum comfort temp
        {'outdoor_temp': 38.0, 'comfort_temp': 32.0},  # Maximum comfort temp
        {'outdoor_temp': 30.0, 'comfort_temp': 40.0},  # Extreme Edge case requested by BJ
    ]
    
    for case in test_cases:
        logging.info(f"\nPrediction for case:")
        logging.info(f"Outdoor Temperature: {case['outdoor_temp']}°C")
        logging.info(f"Desired Comfort Temperature: {case['comfort_temp']}°C")
        
        result = predictor.predict_with_comfort(
            outdoor_temp=case['outdoor_temp'],
            comfort_temp=case['comfort_temp']
        )
        
        if result['success']:
            logging.info("\nPrediction Results:")
            logging.info(f"Calculated PMV: {result['pmv']:.2f}")
            logging.info(f"Thermal Sensation: {result['thermal_sensation']}")
            logging.info(f"Predicted Air Temperature: {result['temperature']:.2f}°C")
            logging.info(f"Predicted Air Velocity: {result['velocity']:.2f} m/s")
        else:
            logging.error(f"Prediction failed: {result['message']}")

if __name__ == "__main__":
    main()