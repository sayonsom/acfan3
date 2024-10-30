import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, concatenate
import joblib
import os
import logging
import json
from typing import Tuple, Optional, Union, List, Dict
from datetime import datetime
from pythermalcomfort.models import pmv_ppd
from tabulate import tabulate
import pandas as pd


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

class FanSpeedConverter:
    """Handles fan speed conversions based on manufacturer specs"""
    def __init__(self, manufacturer: str):
        self.manufacturer = manufacturer
        self.config = self._load_manufacturer_config()

        # Calibration coefficient (derived from empirical data)
        self.K = 0.00099  # Calibration constant
        
        # Blade effect factors
        self.blade_factors = {
            3: 1.0,    # baseline for 3 blades
            4: 1.04,   # 4% increase for 4 blades
            5: 1.07    # 7% increase for 5 blades
        }


    def _load_manufacturer_config(self) -> dict:
        config_path = os.path.join("partners", f"{self.manufacturer}.json")
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Configuration not found for: {self.manufacturer}")

    def calculate_air_velocity(self, rpm: float, distance: float = 2.0) -> float:
        """
        Calculate air velocity using calibrated physics formula
        
        Args:
            rpm: Fan speed in RPM
            distance: Distance from fan in meters
            
        Returns:
            float: Air velocity in m/s
        """
        specs = self.config["specifications"]
        D = specs["sweep_diameter"]  # meters
        
        # Get blade effect factor
        num_blades = specs["blade_count"]
        blade_factor = self.blade_factors.get(num_blades, 1.0)
        
        # Calibrated physics formula
        velocity = (self.K * 3.14159 * D * rpm * blade_factor) / (distance ** 1.8)
        
        return velocity

    def calculate_required_rpm(self, target_velocity: float, distance: float = 2.0) -> float:
        """
        Calculate required RPM using inverse of calibrated formula
        
        Args:
            target_velocity: Desired air velocity in m/s
            distance: Distance from fan in meters
            
        Returns:
            float: Required RPM to achieve target velocity
        """
        specs = self.config["specifications"]
        D = specs["sweep_diameter"]
        
        # Get blade effect factor
        num_blades = specs["blade_count"]
        blade_factor = self.blade_factors.get(num_blades, 1.0)
        
        # Inverse of calibrated formula
        rpm = (target_velocity * (distance ** 1.8)) / (self.K * 3.14159 * D * blade_factor)
        
        return rpm

    def find_closest_speed_setting(self, target_velocity: float, distance: float = 2.0) -> Tuple[Union[int, str], float, float, float]:
        """Find closest speed setting for desired velocity"""
        speed_settings = self.config["speed_settings"]
        min_diff = float('inf')
        best_setting = None
        best_rpm = None
        best_velocity = None
        
        for setting, rpm in speed_settings.items():
            velocity = self.calculate_air_velocity(rpm, distance)
            diff = abs(velocity - target_velocity)
            
            if diff < min_diff:
                min_diff = diff
                best_setting = setting
                best_rpm = rpm
                best_velocity = velocity
        
        # Calculate precise RPM needed
        precise_rpm = self.calculate_required_rpm(target_velocity, distance)
        
        return best_setting, best_rpm, best_velocity, precise_rpm

class ComfortBasedPredictor:
    """Combined comfort-based predictor with fan speed calculations"""
    
    def __init__(
        self,
        model_path: str,
        scaler_x_path: str,
        scaler_y_path: str,
        manufacturer: str,
        temp_bounds: Tuple[float, float] = (18.0, 30.0),
        velocity_bounds: Tuple[float, float] = (0.1, 2.0)
    ):
        """
        Initialize the predictor with model paths and configuration
        
        Args:
            model_path: Path to the saved Keras model
            scaler_x_path: Path to the input scaler
            scaler_y_path: Path to the output scaler
            manufacturer: Name of the fan manufacturer (for config)
            temp_bounds: Valid temperature range (min, max)
            velocity_bounds: Valid velocity range (min, max)
        """
        # Store paths and bounds
        self.model_path = model_path
        self.scaler_x_path = scaler_x_path
        self.scaler_y_path = scaler_y_path
        self.temp_bounds = temp_bounds
        self.velocity_bounds = velocity_bounds
        
        # Initialize model and scalers
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        
        # Initialize fan converter
        self.fan_converter = FanSpeedConverter(manufacturer)
        
        # Input validation bounds
        self.input_bounds = {
            'outdoor_temp': (-10.0, 50.0),
            'comfort_temp': (16.0, 32.0),
            'pmv': (-3.0, 3.0)
        }
        
        # PMV calculation constants
        self.PMV_REFERENCE_TEMP = 24.0
        self.PMV_RATE = 0.3
        
        # Default thermal comfort parameters
        self.default_params = {
            'rh': 50,  # Relative humidity (%)
            'met': 1.0,  # Metabolic rate (met)
            'clo': 0.5,  # Clothing insulation (clo)
            'tr': None  # Mean radiant temperature (°C) - will be set to air temperature
        }
        
        # Load the model and scalers immediately
        self.load()
    
    def load(self) -> bool:
        """Load the model and scalers"""
        try:
            logging.info("Loading model and scalers...")
            
            # Check if files exist
            for path, desc in [
                (self.model_path, "Model"),
                (self.scaler_x_path, "Input scaler"),
                (self.scaler_y_path, "Output scaler")
            ]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{desc} file not found: {path}")
            
            # Load model with custom objects
            custom_objects = {'CustomFeatureLayer': CustomFeatureLayer}
            self.model = load_model(self.model_path, custom_objects=custom_objects)
            
            # Load scalers
            self.scaler_x = joblib.load(self.scaler_x_path)
            self.scaler_y = joblib.load(self.scaler_y_path)
            
            logging.info("Model and scalers loaded successfully!")
            return True
            
        except Exception as e:
            logging.error(f"Error loading model or scalers: {e}")
            return False
    
    def calculate_pmv(self, comfort_temp: float) -> float:
        """
        Calculate PMV based on comfort temperature preference
        
        Args:
            comfort_temp: Desired comfort temperature in Celsius
            
        Returns:
            float: Calculated PMV value
        """
        temp_difference = comfort_temp - self.PMV_REFERENCE_TEMP
        pmv = temp_difference * self.PMV_RATE
        return np.clip(pmv, -3.0, 3.0)
    
    def calculate_pmv_detailed(self, ta: float, vel: float, tr: float = None) -> float:
        """
        Calculate PMV using pythermalcomfort library with detailed parameters
        
        Args:
            ta: Air temperature (°C)
            vel: Air velocity (m/s)
            tr: Mean radiant temperature (°C), defaults to air temperature if None
            
        Returns:
            float: Calculated PMV value
        """
        if tr is None:
            tr = ta
            
        pmv = pmv_ppd(
            tdb=ta,  # air temperature
            tr=tr,   # mean radiant temperature
            vr=vel,  # relative air velocity
            rh=self.default_params['rh'],
            met=self.default_params['met'],
            clo=self.default_params['clo'],
        )
        return pmv['pmv']
    
    def get_pmv_description(self, pmv: float) -> str:
        """
        Get thermal sensation description based on PMV value
        
        Args:
            pmv: PMV value
            
        Returns:
            str: Description of thermal sensation
        """
        if pmv <= -2.5: return "Cold"
        elif pmv <= -1.5: return "Cool"
        elif pmv <= -0.5: return "Slightly Cool"
        elif pmv < 0.5: return "Neutral"
        elif pmv < 1.5: return "Slightly Warm"
        elif pmv < 2.5: return "Warm"
        else: return "Hot"
    
    def validate_input(self, outdoor_temp: float, comfort_temp: float) -> Tuple[bool, str]:
        """
        Validate input temperature values
        
        Args:
            outdoor_temp: Outdoor temperature in Celsius
            comfort_temp: Desired comfort temperature in Celsius
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not self.input_bounds['outdoor_temp'][0] <= outdoor_temp <= self.input_bounds['outdoor_temp'][1]:
            return False, f"Outdoor temperature {outdoor_temp}°C is outside valid range {self.input_bounds['outdoor_temp']}"
        if not self.input_bounds['comfort_temp'][0] <= comfort_temp <= self.input_bounds['comfort_temp'][1]:
            return False, f"Comfort temperature {comfort_temp}°C is outside valid range {self.input_bounds['comfort_temp']}"
        return True, ""
    
    def predict_with_comfort(
        self,
        outdoor_temp: float,
        comfort_temp: float = 24.0,
        distance: float = 2.0,
        validate: bool = True
    ) -> Dict[str, Union[float, str, bool, dict]]:
        """
        Make prediction with comfort temperature and get fan settings
        
        Args:
            outdoor_temp: Outdoor temperature in Celsius
            comfort_temp: Desired comfort temperature in Celsius
            distance: Distance from fan in meters
            validate: Whether to validate inputs
            
        Returns:
            Dictionary containing predictions and settings
        """
        result = {
            'success': False,
            'message': '',
            'inputs': {
                'outdoor_temp': outdoor_temp,
                'comfort_temp': comfort_temp
            },
            'outputs': None
        }
        
        try:
            # Validate input if requested
            if validate:
                is_valid, error_msg = self.validate_input(outdoor_temp, comfort_temp)
                if not is_valid:
                    result['message'] = error_msg
                    return result
            
            # Calculate desired PMV from comfort temperature
            pmv_desired = self.calculate_pmv(comfort_temp)
            
            # Get predictions
            input_data = np.array([[outdoor_temp, pmv_desired]])
            input_scaled = self.scaler_x.transform(input_data)
            predictions_scaled = self.model.predict(input_scaled, verbose=0)
            
            if isinstance(predictions_scaled, list):
                predictions = self.scaler_y.inverse_transform(
                    np.column_stack((predictions_scaled[0], predictions_scaled[1]))
                )
            else:
                predictions = self.scaler_y.inverse_transform(predictions_scaled)
            
            predicted_temp = float(predictions[0, 0])
            predicted_velocity = float(predictions[0, 1])
            
            # Get fan settings
            speed_setting, rpm, actual_velocity, precise_rpm = self.fan_converter.find_closest_speed_setting(
                predicted_velocity, distance
            )
            
            # Calculate delivered PMV using pythermalcomfort
            pmv_delivered = self.calculate_pmv_detailed(
                ta=predicted_temp,
                vel=actual_velocity,
                tr=predicted_temp
            )
            
            result.update({
                'success': True,
                'message': 'Prediction successful',
                'outputs': {
                    'pmv_desired': pmv_desired,
                    'pmv_delivered': pmv_delivered,
                    'air_speed_predicted': predicted_velocity,
                    'air_speed_delivered': actual_velocity,
                    'fan_speed_delivered': speed_setting,
                    'desired_rpm': precise_rpm,
                    'available_rpm': rpm,
                    'predicted_temp': predicted_temp
                }
            })
            
            return result
            
        except Exception as e:
            result['message'] = f"Error during prediction: {str(e)}"
            return result



def main():
    """Example usage with tabular output"""
    MODEL_DIR = 'models'
    SCALER_DIR = 'scalers'
    
    predictor = ComfortBasedPredictor(
        model_path=os.path.join(MODEL_DIR, 'final_model.keras'),
        scaler_x_path=os.path.join(SCALER_DIR, 'scaler_X.joblib'),
        scaler_y_path=os.path.join(SCALER_DIR, 'scaler_y.joblib'),
        manufacturer='polycab'
    )
    
    test_cases = [
        {'outdoor_temp': 32.0, 'comfort_temp': 24.0},  # Reference comfort
        {'outdoor_temp': 35.0, 'comfort_temp': 20.0},  # Cool preference
        {'outdoor_temp': 30.0, 'comfort_temp': 28.0},  # Warm preference
    ]
    
    # Collect results for tabular display
    results_data = []
    
    for case in test_cases:
        result = predictor.predict_with_comfort(outdoor_temp=case['outdoor_temp'],comfort_temp=case['comfort_temp'])
        
        if result['success']:
            
            row = {
                'Outdoor_Temp': case['outdoor_temp'],
                'Comfort_Temp': case['comfort_temp'],
                'PMV_Desired': result['outputs']['pmv_desired'],
                'PMV_Delivered': result['outputs']['pmv_delivered'],
                'Air_Speed_Predicted': result['outputs']['air_speed_predicted'],
                'Air_Speed_Delivered': result['outputs']['air_speed_delivered'],
                'Fan_Speed': result['outputs']['fan_speed_delivered'],
                'Desired_RPM': result['outputs']['desired_rpm'],
                'Available_RPM': result['outputs']['available_rpm']
            }
            results_data.append(row)
    
    # Create DataFrame
    # breakpoint()
    df = pd.DataFrame(results_data)
    
    # Format numeric columns
    format_dict = {
        'PMV_Desired': '{:.2f}',
        'PMV_Delivered': '{:.2f}',
        'Air_Speed_Predicted': '{:.3f}',
        'Air_Speed_Delivered': '{:.3f}',
        'Desired_RPM': '{:.1f}',
        'Available_RPM': '{:.1f}'
    }
    
    # Apply formatting
    for col, format_str in format_dict.items():
        df[col] = df[col].apply(lambda x: format_str.format(float(x)))
    
    # Rename columns for display
    display_names = {
        'Outdoor_Temp': 'Outdoor Temp (°C)',
        'Comfort_Temp': 'Comfort Temp (°C)',
        'PMV_Desired': 'PMV Desired',
        'PMV_Delivered': 'PMV Delivered',
        'Air_Speed_Predicted': 'Air Speed Predicted (m/s)',
        'Air_Speed_Delivered': 'Air Speed Delivered (m/s)',
        'Fan_Speed': 'Fan Speed Setting',
        'Desired_RPM': 'Desired RPM',
        'Available_RPM': 'Available RPM'
    }
    
    df = df.rename(columns=display_names)
    
    # Display results
    print("\nFan Comfort Analysis Results:")
    print("-" * 100)
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    
    # Additional comfort analysis
    print("\nComfort Analysis Summary:")
    print("-" * 100)
    for i, row in df.iterrows():
        pmv_diff = float(row['PMV Delivered']) - float(row['PMV Desired'])
        air_speed_diff = float(row['Air Speed Delivered (m/s)']) - float(row['Air Speed Predicted (m/s)'])
        
        print(f"\nCase {i+1}:")
        print(f"Conditions: {row['Outdoor Temp (°C)']}°C outdoor, {row['Comfort Temp (°C)']}°C desired")
        print(f"PMV Difference: {pmv_diff:+.2f} from desired")
        print(f"Air Speed Difference: {air_speed_diff:+.3f} m/s from predicted")
        print(f"Fan Setting: {row['Fan Speed Setting']} at {row['Available RPM']} RPM")

if __name__ == "__main__":
    main()