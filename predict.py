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

class ACFanPredictor:
    """ACFan predictor for setpoint and air velocity"""
    
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
                pmv_target, metabolic_rate,
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


def generate_sensitivity_ranges():
    """Generate test ranges for each parameter"""
    return {
        'total_occupants': list(range(1, 6)),  # 1 to 5 occupants
        'indoor_temp': np.arange(22.0, 32.0, 1.0),  # 22°C to 31°C
        'outdoor_temp': np.arange(25.0, 45.0, 2.0),  # 25°C to 43°C
        'ac_capacity': [1.0, 1.5, 2.0],  # Common AC capacities
        'humidity': np.arange(30.0, 75.0, 5.0),  # 30% to 70%
        'air_velocity': np.arange(0.1, 1.3, 0.1),  # 0.1 to 1.2 m/s
        'pmv_target': [-1.0, -0.5, 0.0, 0.5, 1.0],  # Common PMV targets
        'metabolic_rate': np.arange(0.8, 2.1, 0.2),  # 0.8 to 2.0 met
        'clothing_insulation': np.arange(0.3, 1.6, 0.1),  # 0.3 to 1.5 clo
        'age': [25, 35, 45, 55, 65]  # Representative age groups
    }

def perform_sensitivity_analysis(predictor: ACFanPredictor) -> pd.DataFrame:
    """
    Perform sensitivity analysis by varying each parameter while keeping others constant
    """
    # Base case (comfortable conditions)
    base_case = {
        'total_occupants': 2,
        'indoor_temp': 26.0,
        'outdoor_temp': 32.0,
        'ac_capacity': 1.5,
        'humidity': 50.0,
        'air_velocity': 0.3,
        'pmv_target': 0.0,
        'metabolic_rate': 1.2,
        'clothing_insulation': 0.5,
        'age': 35
    }
    
    # Get test ranges
    sensitivity_ranges = generate_sensitivity_ranges()
    
    # Store results
    results = []
    
    # Test each parameter
    for param in base_case.keys():
        logging.info(f"\nTesting sensitivity for: {param}")
        test_range = sensitivity_ranges[param]
        
        for test_value in test_range:
            # Create test case with current parameter value
            test_case = base_case.copy()
            test_case[param] = test_value
            
            # Make prediction
            result = predictor.predict(**test_case)
            
            if result['success']:
                results.append({
                    'parameter': param,
                    'test_value': test_value,
                    'setpoint': result['setpoint'],
                    'velocity': result['velocity'],
                    'indoor_temp': test_case['indoor_temp'],
                    'outdoor_temp': test_case['outdoor_temp'],
                    'humidity': test_case['humidity'],
                    'pmv_target': test_case['pmv_target']
                })
            else:
                logging.warning(f"Prediction failed for {param} = {test_value}: {result['message']}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate additional metrics
    df['setpoint_delta'] = df.groupby('parameter')['setpoint'].transform(lambda x: x - x.mean())
    df['velocity_delta'] = df.groupby('parameter')['velocity'].transform(lambda x: x - x.mean())
    
    # Add percentage changes
    df['setpoint_pct_change'] = df.groupby('parameter')['setpoint'].transform(
        lambda x: (x - x.iloc[0]) / x.iloc[0] * 100
    )
    df['velocity_pct_change'] = df.groupby('parameter')['velocity'].transform(
        lambda x: (x - x.iloc[0]) / x.iloc[0] * 100
    )
    
    return df

def save_sensitivity_results(df: pd.DataFrame, output_dir: str = 'sensitivity_results'):
    """Save sensitivity analysis results in multiple formats"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed CSV
    detail_path = os.path.join(output_dir, f'sensitivity_analysis_detail_{timestamp}.csv')
    df.to_csv(detail_path, index=False)
    
    # Create summary DataFrame
    summary_data = []
    for param in df['parameter'].unique():
        param_data = df[df['parameter'] == param]
        summary_data.append({
            'parameter': param,
            'min_setpoint': param_data['setpoint'].min(),
            'max_setpoint': param_data['setpoint'].max(),
            'setpoint_range': param_data['setpoint'].max() - param_data['setpoint'].min(),
            'min_velocity': param_data['velocity'].min(),
            'max_velocity': param_data['velocity'].max(),
            'velocity_range': param_data['velocity'].max() - param_data['velocity'].min(),
            'max_setpoint_pct_change': param_data['setpoint_pct_change'].abs().max(),
            'max_velocity_pct_change': param_data['velocity_pct_change'].abs().max()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f'sensitivity_analysis_summary_{timestamp}.csv')
    summary_df.to_csv(summary_path, index=False)
    
    logging.info(f"\nSensitivity analysis results saved:")
    logging.info(f"Detailed results: {detail_path}")
    logging.info(f"Summary results: {summary_path}")
    
    return detail_path, summary_path

def main():
    """Perform sensitivity analysis"""
    
    # Define paths
    MODEL_DIR = 'models'
    SCALER_DIR = 'scalers'
    
    model_path = os.path.join(MODEL_DIR, 'final_model.keras')
    scaler_x_path = os.path.join(SCALER_DIR, 'scaler_X.joblib')
    scaler_y_path = os.path.join(SCALER_DIR, 'scaler_y.joblib')
    
    # Create predictor instance
    predictor = ACFanPredictor(
        model_path=model_path,
        scaler_x_path=scaler_x_path,
        scaler_y_path=scaler_y_path
    )
    
    # Perform sensitivity analysis
    logging.info("Starting sensitivity analysis...")
    results_df = perform_sensitivity_analysis(predictor)
    
    # Save results
    detail_path, summary_path = save_sensitivity_results(results_df)
    
    # Print some key findings
    logging.info("\nKey Sensitivity Findings:")
    for param in results_df['parameter'].unique():
        param_data = results_df[results_df['parameter'] == param]
        logging.info(f"\nParameter: {param}")
        logging.info(f"Setpoint range: {param_data['setpoint'].min():.2f}°C to {param_data['setpoint'].max():.2f}°C")
        logging.info(f"Velocity range: {param_data['velocity'].min():.2f} to {param_data['velocity'].max():.2f} m/s")

if __name__ == "__main__":
    main()
