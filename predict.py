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
        # Square features
        squares = tf.square(inputs)
        
        # Simple pairwise feature (multiply adjacent features)
        shifted = inputs[:, 1:]
        original = inputs[:, :-1]
        pairwise = original * shifted
        
        # Concatenate original features with engineered features
        return concatenate([inputs, squares, pairwise])
    
    def compute_output_shape(self, input_shape):
        # Original features + squared features + pairwise features
        return (input_shape[0], input_shape[1] * 3 - 1)

    def get_config(self):
        return super(CustomFeatureLayer, self).get_config()

class HVACPredictor:
    """HVAC predictor for setpoint and air velocity"""
    
    def __init__(
        self,
        model_path: str,
        scaler_x_path: str,
        scaler_y_path: str,
        setpoint_bounds: Tuple[float, float] = (18.0, 30.0),
        velocity_bounds: Tuple[float, float] = (0.1, 1.2)
    ):
        self.model_path = model_path
        self.scaler_x_path = scaler_x_path
        self.scaler_y_path = scaler_y_path
        self.setpoint_bounds = setpoint_bounds
        self.velocity_bounds = velocity_bounds
        
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        
        # Input validation bounds
        self.input_bounds = {
            'total_occupants': (1, 10),
            'indoor_temp': (16.0, 32.0),
            'outdoor_temp': (-10.0, 50.0),
            'ac_capacity': (0.5, 3.0),
            'humidity': (20.0, 80.0),
            'air_velocity': (0.1, 1.5),
            'pmv_initial': (-3.0, 3.0),
            'pmv_target': (-3.0, 3.0),
            'metabolic_rate': (0.8, 2.0),
            'clothing_insulation': (0.3, 1.5),
            'age': (18, 80)
        }
        
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
    
    def validate_input(self, input_data: Dict[str, float]) -> Tuple[bool, str]:
        """Validate input values"""
        for key, value in input_data.items():
            if key in self.input_bounds:
                min_val, max_val = self.input_bounds[key]
                if not min_val <= value <= max_val:
                    return False, f"{key} value {value} is outside valid range [{min_val}, {max_val}]"
        return True, ""
    
    def predict(
        self,
        total_occupants: int,
        indoor_temp: float,
        outdoor_temp: float,
        ac_capacity: float,
        humidity: float,
        air_velocity: float,
        pmv_initial: float,
        pmv_target: float,
        metabolic_rate: float,
        clothing_insulation: float,
        age: int,
        validate: bool = True
    ) -> Dict[str, Union[float, str, bool]]:
        """
        Make prediction based on input parameters
        
        Returns:
            Dictionary containing predictions and metadata
        """
        result = {
            'success': False,
            'setpoint': None,
            'velocity': None,
            'message': '',
            'input_valid': True
        }
        
        try:
            # Load model if needed
            if self.model is None:
                if not self.load():
                    result['message'] = "Failed to load model and scalers"
                    return result
            
            # Prepare input data
            input_data = {
                'total_occupants': total_occupants,
                'indoor_temp': indoor_temp,
                'outdoor_temp': outdoor_temp,
                'ac_capacity': ac_capacity,
                'humidity': humidity,
                'air_velocity': air_velocity,
                'pmv_initial': pmv_initial,
                'pmv_target': pmv_target,
                'metabolic_rate': metabolic_rate,
                'clothing_insulation': clothing_insulation,
                'age': age
            }
            
            # Validate input if requested
            if validate:
                is_valid, error_msg = self.validate_input(input_data)
                if not is_valid:
                    result['input_valid'] = False
                    result['message'] = error_msg
                    return result
            
            # Convert to numpy array
            input_array = np.array([[
                total_occupants, indoor_temp, outdoor_temp,
                ac_capacity, humidity, air_velocity,
                pmv_initial, pmv_target, metabolic_rate,
                clothing_insulation, age
            ]])
            
            # Scale input
            input_scaled = self.scaler_x.transform(input_array)
            
            # Make prediction
            predictions_scaled = self.model.predict(input_scaled, verbose=0)
            
            # Process predictions
            if isinstance(predictions_scaled, list):
                predictions = self.scaler_y.inverse_transform(
                    np.column_stack((predictions_scaled[0], predictions_scaled[1]))
                )
            else:
                predictions = self.scaler_y.inverse_transform(predictions_scaled)
            
            # Clip predictions to bounds
            setpoint = np.clip(predictions[0, 0], *self.setpoint_bounds)
            velocity = np.clip(predictions[0, 1], *self.velocity_bounds)
            
            result.update({
                'success': True,
                'setpoint': float(setpoint),
                'velocity': float(velocity),
                'message': 'Prediction successful'
            })
            
            return result
            
        except Exception as e:
            result['message'] = f"Error during prediction: {str(e)}"
            return result

def main():
    """Example usage of the HVACPredictor"""
    
    # Define paths
    MODEL_DIR = 'models'
    SCALER_DIR = 'scalers'
    
    model_path = os.path.join(MODEL_DIR, 'final_model.keras')
    scaler_x_path = os.path.join(SCALER_DIR, 'scaler_X.joblib')
    scaler_y_path = os.path.join(SCALER_DIR, 'scaler_y.joblib')
    
    # Create predictor instance
    predictor = HVACPredictor(
        model_path=model_path,
        scaler_x_path=scaler_x_path,
        scaler_y_path=scaler_y_path
    )
    
    # Example test cases
    test_cases = [
        {
            'total_occupants': 2,
            'indoor_temp': 26.0,
            'outdoor_temp': 32.0,
            'ac_capacity': 1.5,
            'humidity': 50.0,
            'air_velocity': 0.3,
            'pmv_initial': 0.5,
            'pmv_target': 0.0,
            'metabolic_rate': 1.2,
            'clothing_insulation': 0.5,
            'age': 30
        },
        {
            'total_occupants': 4,
            'indoor_temp': 28.0,
            'outdoor_temp': 35.0,
            'ac_capacity': 2.0,
            'humidity': 60.0,
            'air_velocity': 0.4,
            'pmv_initial': 1.0,
            'pmv_target': -0.5,
            'metabolic_rate': 1.4,
            'clothing_insulation': 0.6,
            'age': 45
        }
    ]
    
    for case in test_cases:
        logging.info(f"\nPrediction for case:")
        for key, value in case.items():
            logging.info(f"{key}: {value}")
        
        result = predictor.predict(**case)
        
        if result['success']:
            logging.info("\nPrediction Results:")
            logging.info(f"Predicted Setpoint (MRT): {result['setpoint']:.2f}°C")
            logging.info(f"Predicted Air Velocity: {result['velocity']:.2f} m/s")
        else:
            logging.error(f"Prediction failed: {result['message']}")

if __name__ == "__main__":
    main()