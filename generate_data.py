import numpy as np
import pandas as pd
from pythermalcomfort.models import pmv_ppd
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import warnings

# Random seeding
np.random.seed(2024)

def generate_temperature_cycles(num_samples, hours):
    """
    Generate randomized temperature cycles with daily patterns and variations
    """
    # Calculate the number of days needed
    num_days = int(np.ceil(num_samples / (24 * 4)))  # Round up to ensure enough samples
    
    # Generate daily base temperatures with gradual changes
    outdoor_base_temps = np.random.uniform(30, 40, num_days)  # Base temperature varies between 30-40°C
    indoor_base_temps = np.random.uniform(25, 32, num_days)   # Base temperature varies between 25-32°C
    
    # Smooth the base temperatures using rolling average
    outdoor_base_temps = pd.Series(outdoor_base_temps).rolling(window=3, min_periods=1).mean()
    indoor_base_temps = pd.Series(indoor_base_temps).rolling(window=3, min_periods=1).mean()
    
    # Generate amplitudes for each day
    outdoor_amplitudes = np.random.uniform(8, 12, num_days)  # Daily temperature swing 16-24°C
    indoor_amplitudes = np.random.uniform(3, 6, num_days)    # Daily temperature swing 6-12°C
    
    # Initialize arrays for final temperatures
    outdoor_temps = []
    indoor_temps = []
    
    # Generate temperatures for each day
    for day in range(num_days):
        # Create daily cycle with random phase shift
        phase_shift = np.random.uniform(-2, 2)  # Random shift in peak time
        
        for hour in hours:
            # Calculate outdoor temperature
            outdoor_base = outdoor_base_temps[day]
            outdoor_amplitude = outdoor_amplitudes[day]
            outdoor_temp = outdoor_base + outdoor_amplitude * np.sin((2 * np.pi / 24) * (hour - 14 + phase_shift))
            
            # Add some random noise
            outdoor_temp += np.random.normal(0, 0.5)
            
            # Calculate indoor temperature (lagging outdoor by 2-3 hours)
            indoor_base = indoor_base_temps[day]
            indoor_amplitude = indoor_amplitudes[day]
            lag = np.random.uniform(2, 3)
            indoor_temp = indoor_base + indoor_amplitude * np.sin((2 * np.pi / 24) * (hour - 16 - lag + phase_shift))
            
            # Add some random noise
            indoor_temp += np.random.normal(0, 0.3)
            
            # Repeat each hour's temperature 4 times (15-minute intervals)
            outdoor_temps.extend([outdoor_temp] * 4)
            indoor_temps.extend([indoor_temp] * 4)
    
    # Clip temperatures to desired ranges and ensure exact number of samples
    outdoor_temps = np.array(outdoor_temps[:num_samples])
    indoor_temps = np.array(indoor_temps[:num_samples])
    
    outdoor_temps = np.clip(outdoor_temps, 25, 50)
    indoor_temps = np.clip(indoor_temps, 22, 40)
    
    return outdoor_temps, indoor_temps

# Constants for simulation
num_samples = 1_000  # 1000 data points
relative_humidity = 50  # Constant 50%
metabolic_rate = 1.0   # Met = 1.0 (moderate physical activity)
clothing_insulation = 0.5  # Clo = 0.5 (typical indoor clothing)

# Generate hours array
hours = np.arange(0, 24, 1)

# Generate temperatures
print("Generating temperature patterns...")
outdoor_temp, indoor_temp = generate_temperature_cycles(num_samples, hours)

# Verify array sizes
print(f"\nArray sizes after generation:")
print(f"Outdoor temperature array size: {len(outdoor_temp)}")
print(f"Indoor temperature array size: {len(indoor_temp)}")

# Generate timestamps
timestamps = [datetime(2024, 1, 1) + timedelta(minutes=15 * i) for i in range(num_samples)]

# Generate Air Velocity with adjusted log-normal distribution
target_air_velocity = 0.5  
mu = np.log(target_air_velocity)  # Most likely air speed inside the room by turning on the fan to 
sigma = 0.1       
air_velocity = np.random.lognormal(mean=mu, sigma=sigma, size=num_samples)
air_velocity = np.clip(air_velocity, 0.1, 0.8)

# Calculate PMV values
print("\nCalculating PMV values...")
pmv_values = []
cooling_warnings = []
ppd_values = []

for i in range(num_samples):
    if i % 200 == 0:  # Adjusted progress reporting for smaller dataset
        print(f"Processing sample {i}/{num_samples}")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        result = pmv_ppd(
            tdb=indoor_temp[i],
            tr=indoor_temp[i],
            vr=air_velocity[i],
            rh=relative_humidity,
            met=metabolic_rate,
            clo=clothing_insulation,
            standard='ashrae',
            limit_inputs=True
        )
        
        pmv_values.append(result['pmv'])
        ppd_values.append(result['ppd'])
        cooling_warnings.append(any("cooling effect" in str(warn.message).lower() for warn in w))

# Create DataFrame
df = pd.DataFrame({
    'Timestamp': timestamps,
    'Outdoor_Temperature': outdoor_temp,
    'Air_Temperature': indoor_temp,
    'Air_Velocity': air_velocity,
    'PMV': np.clip(pmv_values, -3, 3),
    'PPD': ppd_values,
    'Cooling_Warning': cooling_warnings
})

# Save dataset
df.to_csv('synthetic_ac_fan_data.csv', index=False)
print("\nDataset saved to 'synthetic_ac_fan_data.csv'")

# Plot distributions
plt.figure(figsize=(15, 10))

# Plot 1: Temperature distributions
plt.subplot(2, 2, 1)
plt.hist(outdoor_temp, bins=50, alpha=0.5, label='Outdoor', density=True)
plt.hist(indoor_temp, bins=50, alpha=0.5, label='Indoor', density=True)
plt.xlabel('Temperature (°C)')
plt.ylabel('Density')
plt.title('Temperature Distributions')
plt.legend()
plt.grid(True)

# Plot 2: Temperature patterns for first 24 hours
plt.subplot(2, 2, 2)
plt.plot(outdoor_temp[:96], label='Outdoor', alpha=0.7)
plt.plot(indoor_temp[:96], label='Indoor', alpha=0.7)
plt.xlabel('15-minute intervals')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Patterns (First 24 Hours)')
plt.legend()
plt.grid(True)

# Plot 3: Air Velocity Distribution
plt.subplot(2, 2, 3)
plt.hist(air_velocity, bins=50, density=True)
plt.xlabel('Air Velocity (m/s)')
plt.ylabel('Density')
plt.title('Air Velocity Distribution')
plt.grid(True)

# Plot 4: PMV Distribution
plt.subplot(2, 2, 4)
plt.hist(df['PMV'], bins=50, density=True)
plt.xlabel('PMV Values')
plt.ylabel('Density')
plt.title('PMV Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nFinal Dataset Statistics:")
print(df.describe())

# Print warning statistics
warning_count = df['Cooling_Warning'].sum()
warning_percentage = (warning_count / num_samples) * 100
print(f"\nCooling warnings occurred in {warning_percentage:.2f}% of samples")