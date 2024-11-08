import numpy as np
import pandas as pd
from pythermalcomfort.models import pmv_ppd
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(2024)

def suppress_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 
                                  message='Assuming cooling effect = 0.*',
                                  category=UserWarning)
            return func(*args, **kwargs)
    return wrapper



def find_mrt_for_pmv_range(
    tdb: float,  # air temperature, °C
    rh: float,   # relative humidity, %
    vr: float,   # relative air velocity, m/s
    met: float,  # metabolic rate, met
    clo: float,  # clothing insulation, clo
    target_pmv_min: float = -0.2,
    target_pmv_max: float = 0.2,
    mrt_min: float = 0,
    mrt_max: float = 50,
    tolerance: float = 0.01
) -> Tuple[float, float]:
    """
    Find the MRT (Mean Radiant Temperature) that results in a PMV within the target range.
    Uses binary search to efficiently find the solution.
    """
    @suppress_warnings
    def calculate_pmv(mrt: float) -> float:
        results = pmv_ppd(tdb=tdb, tr=mrt, rh=rh, vr=vr, met=met, clo=clo)
        return results['pmv']
    
    # First check if solution exists within our range
    pmv_at_min = calculate_pmv(mrt_min)
    pmv_at_max = calculate_pmv(mrt_max)
    
    if pmv_at_min > target_pmv_max and pmv_at_max > target_pmv_max:
        raise ValueError(f"No solution exists in range - PMV always too high (min: {pmv_at_min:.3f}, max: {pmv_at_max:.3f})")
    if pmv_at_min < target_pmv_min and pmv_at_max < target_pmv_min:
        raise ValueError(f"No solution exists in range - PMV always too low (min: {pmv_at_min:.3f}, max: {pmv_at_max:.3f})")
    
    left = mrt_min
    right = mrt_max
    iterations = 0
    max_iterations = 100  # prevent infinite loops
    
    while (right - left) > tolerance and iterations < max_iterations:
        mrt = (left + right) / 2
        current_pmv = calculate_pmv(mrt)
        
        # If PMV is within target range, we found a solution
        if target_pmv_min <= current_pmv <= target_pmv_max:
            return mrt, current_pmv, iterations
        
        # If PMV is too high, decrease MRT by searching in lower half
        elif current_pmv > target_pmv_max:
            right = mrt
        # If PMV is too low, increase MRT by searching in upper half
        else:
            left = mrt
            
        iterations += 1
    
    # Return best found solution
    final_mrt = (left + right) / 2
    final_pmv = calculate_pmv(final_mrt)
    return final_mrt, final_pmv, iterations

@suppress_warnings
def explore_mrt_range(conditions: dict, mrt_range: np.ndarray):
    """Explore PMV values across a range of MRT temperatures"""
    pmv_values = []
    for mrt in mrt_range:
        result = pmv_ppd(tr=mrt, **conditions)
        pmv_values.append(result['pmv'])
    return pmv_values

def get_season(date: datetime) -> str:
    """
    Determine season based on date for Indian climate
    
    Parameters:
    - date: datetime object
    
    Returns:
    - str: season name
    """
    month = date.month
    if 3 <= month <= 6:
        return 'summer'
    elif 7 <= month <= 9:
        return 'monsoon'
    elif 10 <= month <= 11:
        return 'post-monsoon'
    else:
        return 'winter'

def get_time_block(hour: int) -> str:
    """
    Categorize time of day into 4-hour blocks
    
    Parameters:
    - hour: Hour of day (0-23)
    
    Returns:
    - str: time block category
    """
    if 0 <= hour < 4:
        return 'late_night'
    elif 4 <= hour < 8:
        return 'early_morning'
    elif 8 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 16:
        return 'afternoon'
    elif 16 <= hour < 20:
        return 'evening'
    else:
        return 'night'

def generate_temperature_cycles(timestamps: List[datetime]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate temperature patterns considering seasonal transitions
    
    Parameters:
    - timestamps: List of datetime objects
    
    Returns:
    - Tuple of outdoor and indoor temperatures
    """
    num_samples = len(timestamps)
    outdoor_temps = np.zeros(num_samples)
    indoor_temps = np.zeros(num_samples)
    
    # Temperature ranges by season
    temp_ranges = {
        'summer': {'outdoor': (30, 45), 'indoor': (25, 35), 'outdoor_amp': (8, 15), 'indoor_amp': (3, 8)},
        'monsoon': {'outdoor': (25, 35), 'indoor': (24, 32), 'outdoor_amp': (5, 10), 'indoor_amp': (2, 6)},
        'post-monsoon': {'outdoor': (20, 30), 'indoor': (22, 30), 'outdoor_amp': (7, 12), 'indoor_amp': (3, 7)},
        'winter': {'outdoor': (10, 25), 'indoor': (15, 28), 'outdoor_amp': (10, 18), 'indoor_amp': (5, 10)}
    }
    
    # Process day by day
    current_date = timestamps[0].date()
    day_indices = []
    
    # Group indices by day
    for i, ts in enumerate(timestamps):
        if ts.date() != current_date:
            day_indices.append(i)
            current_date = ts.date()
    day_indices.append(len(timestamps))
    
    start_idx = 0
    for end_idx in day_indices:
        if end_idx == start_idx:
            continue
            
        day_timestamps = timestamps[start_idx:end_idx]
        season = get_season(day_timestamps[0])
        ranges = temp_ranges[season]
        
        # Generate base temperatures
        outdoor_base = np.random.uniform(ranges['outdoor'][0], ranges['outdoor'][1])
        indoor_base = np.random.uniform(ranges['indoor'][0], ranges['indoor'][1])
        
        # Generate amplitudes
        outdoor_amp = np.random.uniform(ranges['outdoor_amp'][0], ranges['outdoor_amp'][1])
        indoor_amp = np.random.uniform(ranges['indoor_amp'][0], ranges['indoor_amp'][1])
        
        # Generate temperatures for each timestamp
        for i, ts in enumerate(day_timestamps):
            hour = ts.hour + ts.minute/60
            
            # Calculate outdoor temperature
            outdoor_temp = outdoor_base + outdoor_amp * np.sin((2 * np.pi / 24) * (hour - 14))
            outdoor_temp += np.random.normal(0, 0.5)
            
            # Calculate indoor temperature with lag
            lag = np.random.uniform(3, 4)
            indoor_temp = indoor_base + indoor_amp * np.sin((2 * np.pi / 24) * (hour - 16 - lag))
            indoor_temp += np.random.normal(0, 0.3)
            
            idx = start_idx + i
            outdoor_temps[idx] = np.clip(outdoor_temp, ranges['outdoor'][0], ranges['outdoor'][1])
            indoor_temps[idx] = np.clip(indoor_temp, ranges['indoor'][0], ranges['indoor'][1])
        
        start_idx = end_idx
    
    return outdoor_temps, indoor_temps

def generate_starting_temperature(indoor_temps: np.ndarray) -> np.ndarray:
    """
    Generate starting indoor temperatures with normally distributed difference from indoor temperature
    
    Parameters:
    - indoor_temps: Array of indoor temperatures without AC
    
    Returns:
    - np.ndarray: Array of starting temperatures
    """
    # Generate temperature differences using truncated normal distribution
    temp_differences = np.random.normal(2.5, 0.5, len(indoor_temps))  # mean=2.5, std=0.5
    temp_differences = np.clip(temp_differences, 1, 4)  # Ensure differences are between 1-4 degrees
    print(temp_differences)
    # Calculate starting temperatures
    # sometimes make the starting temperature higher than the indoor temperature and sometimes lower
    starting_temps = indoor_temps - temp_differences
    return np.clip(starting_temps, 24, 30)  # Maintain original bounds

def generate_ac_capacity(num_samples: int) -> np.ndarray:
    """
    Generate AC capacity (tons) based on typical Indian bedroom sizes
    
    Parameters:
    - num_samples: Number of samples to generate
    
    Returns:
    - np.ndarray: Array of AC capacities
    """
    capacities = [1.0, 1.5, 2.0]
    probabilities = [0.25, 0.60, 0.15]
    return np.random.choice(capacities, size=num_samples, p=probabilities)

def generate_humidity(timestamps: List[datetime]) -> np.ndarray:
    """
    Generate humidity based on season and time with skewed distribution between 20-65%
    
    Parameters:
    - timestamps: List of datetime objects
    
    Returns:
    - np.ndarray: Array of humidity values
    """
    num_samples = len(timestamps)
    humidity = np.zeros(num_samples)
    
    # Using beta distribution to create skewed values
    a, b = 3.0, 1.5  # Shape parameters for beta distribution - increased a for more concentration in 50-60 range
    
    for i, ts in enumerate(timestamps):
        # Generate base humidity using beta distribution
        base_humidity = np.random.beta(a, b)
        # Scale to desired range (20-60), leaving room for occupancy effects
        base_humidity = 20 + (base_humidity * 40)  # Scales to 20-60 range
        
        # Add small time-based variation
        hour = ts.hour + ts.minute/60
        time_variation = -3 * np.sin((2 * np.pi / 24) * (hour - 14))  # Reduced variation
        
        humidity[i] = np.clip(base_humidity + time_variation, 20, 60)  # Clip to slightly lower max
    
    return humidity

def generate_air_velocity(timestamps: List[datetime]) -> np.ndarray:
    """
    Generate air velocity based on time and season
    
    Parameters:
    - timestamps: List of datetime objects
    
    Returns:
    - np.ndarray: Array of air velocity values
    """
    num_samples = len(timestamps)
    velocity = np.zeros(num_samples)
    
    velocity_params = {
        'summer': {'mean': 0.2, 'sigma': 0.025},
        'monsoon': {'mean': 0.1, 'sigma': 0.03},
        'post-monsoon': {'mean': 0.05, 'sigma': 0.02},
        'winter': {'mean': 0.01, 'sigma': 0.015}
    }
    
    for i, ts in enumerate(timestamps):
        season = get_season(ts)
        params = velocity_params[season]
        
        base_velocity = np.random.lognormal(mean=np.log(params['mean']), sigma=params['sigma'])
        
        if 9 <= ts.hour < 22:
            base_velocity *= 1.2
            
        velocity[i] = np.clip(base_velocity, 0.1, 1.5)
    
    return velocity

def generate_occupancy_pattern(timestamps: List[datetime], room_size: int = 5) -> np.ndarray:
    """
    Generate realistic room occupancy patterns with predominantly 2 people at night
    
    Parameters:
    - timestamps: List of datetime objects
    - room_size: Maximum room occupancy
    
    Returns:
    - np.ndarray: Array of occupancy values
    """
    num_samples = len(timestamps)
    occupancy = np.zeros(num_samples)
    hours = np.array([ts.hour for ts in timestamps])
    
    for i, hour in enumerate(hours):
        if 22 <= hour or hour < 6:
            prob = [0.02, 0.03, 0.90, 0.03, 0.01, 0.01]
        elif 6 <= hour < 9:
            prob = [0.05, 0.15, 0.50, 0.20, 0.05, 0.05]
        elif 9 <= hour < 17:
            prob = [0.15, 0.25, 0.20, 0.20, 0.15, 0.05]
        elif 17 <= hour < 20:
            prob = [0.05, 0.15, 0.30, 0.25, 0.15, 0.10]
        else:
            prob = [0.03, 0.07, 0.70, 0.10, 0.05, 0.05]
            
        occupancy[i] = np.random.choice(range(room_size + 1), p=prob)
    
    occupancy = pd.Series(occupancy).rolling(window=2, min_periods=1).mean()
    return np.round(occupancy).astype(int)

def generate_age_groups(num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate age groups and specific ages
    
    Parameters:
    - num_samples: Number of samples to generate
    
    Returns:
    - Tuple of age groups and specific ages
    """
    age_probabilities = [0.4, 0.4, 0.2]
    age_groups = np.random.choice(
        ['young', 'middle_aged', 'elderly'],
        size=num_samples,
        p=age_probabilities
    )
    
    ages = np.zeros(num_samples)
    for i, group in enumerate(age_groups):
        if group == 'young':
            ages[i] = np.random.uniform(18, 35)
        elif group == 'middle_aged':
            ages[i] = np.random.uniform(36, 55)
        else:
            ages[i] = np.random.uniform(56, 80)
    
    return age_groups, ages

def generate_metabolic_rates(timestamps: List[datetime], age_groups: np.ndarray) -> np.ndarray:
    """
    Generate metabolic rates based on time of day and age
    
    Parameters:
    - timestamps: List of datetime objects
    - age_groups: Array of age group categories
    
    Returns:
    - np.ndarray: Array of metabolic rates
    """
    num_samples = len(timestamps)
    met_rates = np.zeros(num_samples)
    
    def adjust_metabolic_rate_for_age(base_rate: float, age_group: str) -> float:
        if age_group == 'young':
            return base_rate * np.random.uniform(0.95, 1.05)
        elif age_group == 'middle_aged':
            return base_rate * np.random.uniform(0.90, 1.00)
        else:
            return base_rate * np.random.uniform(0.80, 0.95)
    
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        time_block = get_time_block(hour)
        
        if time_block == 'early_morning':
            base_met = np.random.uniform(0.8, 0.9)
        elif time_block in ['morning', 'afternoon']:
            base_met = np.random.uniform(1.0, 1.1)
        elif time_block == 'evening':
            base_met = np.random.uniform(0.9, 1.0)
        else:
            base_met = np.random.uniform(0.7, 1.0)
        
        met_rates[i] = adjust_metabolic_rate_for_age(base_met, age_groups[i])
        met_rates[i] += np.random.normal(0, 0.05)
    
    return np.clip(met_rates, 0.7, 2.5)

def generate_clothing_insulation(timestamps: List[datetime]) -> np.ndarray:
    """
    Generate clothing insulation values
    
    Parameters:
    - timestamps: List of datetime objects
    
    Returns:
    - np.ndarray: Array of clothing insulation values
    """
    num_samples = len(timestamps)
    clo = np.zeros(num_samples)
    
    for i, ts in enumerate(timestamps):
        season = get_season(ts)
        if season == 'summer':
            base_clo = np.random.uniform(0.4, 0.6)
        elif season == 'monsoon':
            base_clo = np.random.uniform(0.5, 0.7)
        elif season == 'post-monsoon':
            base_clo = np.random.uniform(0.6, 0.9)
        else:
            base_clo = np.random.uniform(0.7, 1.1)
        
        time_block = get_time_block(ts.hour)
        if time_block in ['early_morning', 'night']:
            base_clo += 0.1
        elif time_block in ['morning', 'afternoon']:
            base_clo += 0.2
            
        clo[i] = np.clip(base_clo + np.random.normal(0, 0.05), 0.4, 1.2)
    
    return clo

def generate_room_volumes(num_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate room volumes and available volumes, and corresponding AC capacities
    
    Parameters:
    - num_samples: Number of samples to generate
    
    Returns:
    - Tuple of room volumes, available volumes, and AC capacities
    """
    # Generate room volumes with normal distribution (mean=36, bounded between 30-40)
    room_volumes = np.random.normal(36, 2, num_samples)
    room_volumes = np.clip(room_volumes, 30, 40)
    
    # Generate available volume percentages (log-normally distributed around 92%)
    # Using log-normal to get right-skewed distribution between 75-98%
    mu = np.log(0.92)  # Setting mean to 92%
    sigma = 0.1  # Adjust this to control spread
    available_percentages = np.random.lognormal(mu, sigma, num_samples)
    available_percentages = np.clip(available_percentages, 0.75, 0.98)
    
    # Calculate available volumes
    available_volumes = room_volumes * available_percentages
    
    # Determine AC capacities based on room volumes
    ac_capacities = np.ones(num_samples)  # Default to 1 ton
    ac_capacities[room_volumes >= 32] = 1.5  # 32-37 m³ gets 1.5 ton
    ac_capacities[room_volumes > 37] = 2.0   # >37 m³ gets 2 ton
    
    return room_volumes, available_volumes, ac_capacities


def calculate_room_effects(base_temp: float, base_humidity: float, 
                         num_occupants: int, room_volume: float) -> Tuple[float, float, float]:
    """
    Calculate room temperature and humidity adjustments based on occupants
    
    Parameters:
    - base_temp: Base room temperature
    - base_humidity: Base relative humidity
    - num_occupants: Number of occupants
    - room_volume: Room volume in m³
    
    Returns:
    - Tuple of adjusted temperature, humidity, and velocity factor
    """
    heat_gain_per_person = 100  # Watts
    air_specific_heat = 1005    # J/kg·K
    air_density = 1.225        # kg/m³
    
    delta_T = (heat_gain_per_person * num_occupants * 3600) / (air_specific_heat * air_density * room_volume)
    new_temp = base_temp + min(delta_T, 3)
    
    # humidity calculation
    water_vapor_per_person = 10  # Assumption
    room_air_mass = room_volume * air_density
    humidity_increase = (water_vapor_per_person * num_occupants) / room_air_mass * 5  # Reduced multiplier from 10 to 5
    new_humidity = np.clip(base_humidity + humidity_increase, 20, 65)  # Explicitly clip to desired range
    
    velocity_factor = 1 + (num_occupants - 1) * 0.1
    
    return new_temp, new_humidity, velocity_factor

def simulate_thermal_comfort(timestamps: List[datetime], room_size: int = 5, 
                           base_room_volume: float = 36) -> Dict[str, Any]:
    """
    Simulate thermal comfort for multiple occupants with room volume variations
    """
    num_samples = len(timestamps)
    
    # Generate room volumes and AC capacities
    room_volumes, available_volumes, ac_capacities = generate_room_volumes(num_samples)
    
    # Generate base environmental conditions
    outdoor_temp, base_indoor_temp = generate_temperature_cycles(timestamps)
    base_humidity = generate_humidity(timestamps)
    base_air_velocity = generate_air_velocity(timestamps)
    

    temp_differences = np.random.normal(0, 2, len(base_indoor_temp))
    starting_temps = base_indoor_temp + temp_differences


    
    occupancy = generate_occupancy_pattern(timestamps, room_size)
    
    # Generate age groups for potential occupants
    age_groups, ages = generate_age_groups(num_samples * room_size)
    age_groups = age_groups.reshape(num_samples, room_size)
    ages = ages.reshape(num_samples, room_size)
    
    # Generate base metabolic rates and clothing
    met_rates = generate_metabolic_rates(timestamps * room_size, age_groups.flatten())
    met_rates = met_rates.reshape(num_samples, room_size)
    clothing = generate_clothing_insulation(timestamps)
    
    # Initialize arrays for comfort metrics
    indoor_temps = np.zeros((num_samples, room_size))
    indoor_temps_no_ac = np.zeros((num_samples, room_size))
    humidities = np.zeros((num_samples, room_size))
    air_velocities = np.zeros((num_samples, room_size))
    pmv_values = np.zeros((num_samples, room_size))
    ppd_values = np.zeros((num_samples, room_size))
    
    print("Simulating thermal comfort for multiple occupants...")
    for i in range(num_samples):
        if i % (num_samples // 10) == 0:
            print(f"Processing {i/num_samples*100:.1f}% complete...")
        
        curr_occupants = occupancy[i]
        if curr_occupants > 0:
            # Use actual room volume for calculations
            adjusted_volume = available_volumes[i]
            
            # Calculate room effects
            adj_temp, adj_humidity, vel_factor = calculate_room_effects(
                starting_temps[i],
                base_humidity[i],
                curr_occupants,
                adjusted_volume  # Use available volume for calculations
            )
            
            # Store the no-AC temperature
            no_ac_temp = base_indoor_temp[i]
            
            # Apply adjustments to occupied positions
            for j in range(curr_occupants):
                indoor_temps[i,j] = adj_temp
                indoor_temps_no_ac[i,j] = no_ac_temp
                humidities[i,j] = adj_humidity
                air_velocities[i,j] = base_air_velocity[i] * vel_factor
                
                # Calculate PMV
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = pmv_ppd(
                        tdb=adj_temp,
                        tr=adj_temp,
                        vr=air_velocities[i,j],
                        rh=adj_humidity,
                        met=met_rates[i,j],
                        clo=clothing[i],
                        standard='ashrae',
                        limit_inputs=True
                    )
                    pmv_values[i,j] = result['pmv']
                    ppd_values[i,j] = result['ppd']
    
    return {
        'timestamps': timestamps,
        'occupancy': occupancy,
        'indoor_temps': indoor_temps,
        'indoor_temps_no_ac': indoor_temps_no_ac,
        'humidities': humidities,
        'air_velocities': air_velocities,
        'pmv_values': pmv_values,
        'ppd_values': ppd_values,
        'age_groups': age_groups,
        'ages': ages,
        'outdoor_temp': outdoor_temp,
        'starting_temps': starting_temps,
        'ac_capacities': ac_capacities,
        'room_volumes': room_volumes,
        'available_volumes': available_volumes,
        'met_rates': met_rates,
        'clothing': clothing
    }

def create_analysis_plots(df: pd.DataFrame) -> None:
    """
    Create comprehensive analysis plots
    
    Parameters:
    - df: DataFrame containing simulation results
    """
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Starting vs Indoor Temperature
    plt.subplot(3, 2, 1)
    plt.scatter(df['Starting_Temperature'], df['Indoor_Temperature'], 
               c=df['Total_Occupants'], cmap='viridis', alpha=0.5)
    plt.colorbar(label='Number of Occupants')
    plt.xlabel('Starting Temperature (°C)')
    plt.ylabel('Final Indoor Temperature (°C)')
    plt.title('Starting vs Final Indoor Temperature')
    plt.grid(True)
    
    # Plot 2: PMV by AC Capacity
    plt.subplot(3, 2, 2)
    sns.boxplot(data=df, x='AC_Capacity', y='PMV', hue='Total_Occupants')
    plt.xlabel('AC Capacity (Tons)')
    plt.ylabel('PMV')
    plt.title('PMV Distribution by AC Capacity and Occupancy')
    
    # Plot 3: Temperature Rise by AC Capacity
    plt.subplot(3, 2, 3)
    temp_rise = df['Indoor_Temperature'] - df['Starting_Temperature']
    sns.boxplot(data=df, x='AC_Capacity', y=temp_rise)
    plt.xlabel('AC Capacity (Tons)')
    plt.ylabel('Temperature Rise (°C)')
    plt.title('Temperature Rise by AC Capacity')
    
    # Plot 4: Starting Temperature Distribution by Time
    plt.subplot(3, 2, 4)
    sns.boxplot(data=df, x='Hour', y='Starting_Temperature')
    plt.xlabel('Hour of Day')
    plt.ylabel('Starting Temperature (°C)')
    plt.title('Starting Temperature Distribution by Hour')
    
    # Plot 5: Occupancy Pattern
    plt.subplot(3, 2, 5)
    hourly_occupancy = df.groupby('Hour')['Total_Occupants'].mean()
    plt.plot(hourly_occupancy.index, hourly_occupancy.values)
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Occupancy')
    plt.title('Average Daily Occupancy Pattern')
    plt.grid(True)
    
    # Plot 6: PMV Distribution by Time and Occupancy
    plt.subplot(3, 2, 6)
    pivot_data = df.pivot_table(
        values='PMV',
        index='Hour',
        columns='Total_Occupants',
        aggfunc='mean'
    )
    sns.heatmap(pivot_data, cmap='RdYlBu_r', center=0)
    plt.xlabel('Number of Occupants')
    plt.ylabel('Hour of Day')
    plt.title('Average PMV by Hour and Occupancy')
    
    plt.tight_layout()
    plt.show()


def find_comfort_parameters(
    tdb: float,  # air temperature, °C
    rh: float,   # relative humidity, %
    vr: float,   # initial relative air velocity, m/s
    met: float,  # metabolic rate, met
    clo: float,  # clothing insulation, clo
    min_mrt: float = 26.0,  # minimum acceptable MRT
    target_pmv_min: float = -1,
    target_pmv_max: float = 1,
    mrt_min: float = 16,
    mrt_max: float = 32,
    vr_min: float = 0.1,
    vr_max: float = 1.2,
    max_iterations: int = 200
) -> Tuple[float, float, float, float, int]:
    """
    Enhanced comfort parameter search that guarantees reaching target PMV range.
    Strategy:
    1. Try combinations of higher AC temp with varying air velocity
    2. If that fails, gradually reduce AC temp with maximum air velocity
    """
    def calculate_pmv(mrt: float, air_velocity: float) -> float:
        try:
            results = pmv_ppd(tdb=tdb, tr=mrt, rh=rh, vr=air_velocity, met=met, clo=clo)
            return results['pmv']
        except Exception as e:
            print(f"PMV calculation error for MRT={mrt}, velocity={air_velocity}: {e}")
            return float('inf')

    def find_ideal_base_mrt() -> float:
        """Find ideal MRT with no air velocity"""
        left = mrt_min
        right = mrt_max
        best_mrt = mrt_min
        best_pmv = float('inf')
        
        for _ in range(50):
            mrt = (left + right) / 2
            current_pmv = calculate_pmv(mrt, 0.0)
            
            if abs(current_pmv) < abs(best_pmv):
                best_mrt = mrt
                best_pmv = current_pmv
            
            if target_pmv_min <= current_pmv <= target_pmv_max:
                return mrt
            elif current_pmv > target_pmv_max:
                right = mrt
            else:
                left = mrt
        
        return best_mrt

    def try_find_solution_with_velocity(mrt: float) -> Tuple[float, float, bool]:
        """Try to find solution by adjusting velocity at given MRT"""
        velocities = np.linspace(vr_min, vr_max, 20)  # Test 20 different velocities
        for test_vel in velocities:
            pmv = calculate_pmv(mrt, test_vel)
            if target_pmv_min <= pmv <= target_pmv_max:
                return test_vel, pmv, True
        return vr_max, calculate_pmv(mrt, vr_max), False

    # Phase 1: Find ideal base MRT and starting point
    ideal_base_mrt = find_ideal_base_mrt()
    current_mrt = min(ideal_base_mrt + 4, mrt_max)
    iterations = 0
    
    # Phase 2: Try combinations of higher AC temp with varying air velocity
    while iterations < max_iterations/2 and current_mrt >= min_mrt:
        velocity, pmv, success = try_find_solution_with_velocity(current_mrt)
        
        if success:
            # print(f"Solution found with combined approach after {iterations} iterations:")
            # print(f"MRT: {current_mrt:.2f}°C, Velocity: {velocity:.2f} m/s, PMV: {pmv:.2f}")
            return current_mrt, current_mrt, velocity, pmv, iterations
        
        current_mrt -= 0.25  # Small temperature step
        iterations += 1
    
    # Phase 3: If no solution found, start from current temperature with max velocity
    # and decrease temperature until target PMV is reached
    # print("Starting final phase with maximum air velocity...")
    current_mrt = tdb  # Start from current air temperature
    while iterations < max_iterations:
        pmv = calculate_pmv(current_mrt, vr_max)
        
        if target_pmv_min <= pmv <= target_pmv_max:
            print(f"Solution found with maximum velocity after {iterations} iterations:")
            print(f"MRT: {current_mrt:.2f}°C, Velocity: {vr_max:.2f} m/s, PMV: {pmv:.2f}")
            return current_mrt, current_mrt, vr_max, pmv, iterations
        
        if pmv > target_pmv_max:  # Still too warm
            current_mrt = max(current_mrt - 0.25, mrt_min)
            if current_mrt == mrt_min and pmv > target_pmv_max:
                # Force a solution at minimum MRT and maximum velocity
                min_pmv = calculate_pmv(mrt_min, vr_max)
                print(f"Forced solution at minimum MRT after {iterations} iterations:")
                print(f"MRT: {mrt_min:.2f}°C, Velocity: {vr_max:.2f} m/s, PMV: {min_pmv:.2f}")
                return mrt_min, mrt_min, vr_max, min_pmv, iterations
        else:  # Too cool (shouldn't happen with proper initial temperature)
            current_mrt = min(current_mrt + 0.25, mrt_max)
        
        iterations += 1
    
    # This point should never be reached with proper ranges
    final_pmv = calculate_pmv(mrt_min, vr_max)
    # print(f"Warning: Using minimum MRT and maximum velocity after {iterations} iterations:")
    # print(f"MRT: {mrt_min:.2f}°C, Velocity: {vr_max:.2f} m/s, PMV: {final_pmv:.2f}")
    return mrt_min, mrt_min, vr_max, final_pmv, iterations


def select_elderly_pmv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select PMV values based on oldest occupant at each timestamp and calculate comfort parameters
    """
    # First identify the oldest person for each timestamp
    oldest_per_timestamp = df.groupby('Timestamp').apply(
        lambda x: x.loc[x['Age'].idxmax()]
    ).reset_index(drop=True)
    
    # Calculate comfort parameters for each row
    comfort_params = []
    for _, row in oldest_per_timestamp.iterrows():
        mrt, feeling_mrt, desired_vr, target_pmv, _ = find_comfort_parameters(
            tdb=row['Actual_Indoor_Temp'],
            rh=row['Humidity'],
            vr=row['Air_Velocity'],
            met=row['Metabolic_Rate'],
            clo=row['Clothing_Insulation']
        )
        comfort_params.append({
            'AC_Setpoint(MRT)': mrt,
            # 'Feeling_MRT': feeling_mrt,
            'Desired_AirVelocity': desired_vr,
            'PMV_Target': target_pmv
        })
    
    # Add comfort parameters to the dataframe
    comfort_df = pd.DataFrame(comfort_params)
    final_df = pd.concat([oldest_per_timestamp, comfort_df], axis=1)
    
    # Rename original PMV column to PMV_Initial
    final_df = final_df.rename(columns={'PMV': 'PMV_Initial'})

    # Make the AC_Setpoint(MRT) column rounded to lower integer
    final_df['AC_Setpoint(MRT)'] = final_df['AC_Setpoint(MRT)'].apply(np.floor)

    # Round the Desired_AirVelocity to 2 decimal places
    final_df['Desired_AirVelocity'] = final_df['Desired_AirVelocity'].round(2)
    
    # Select and order columns as requested
    columns = ['Total_Occupants', 
              'Actual_Indoor_Temp', 'Estimate_Indoor_Temp_due_to_Outdoor', 
              'Outdoor_Temperature', 'AC_Capacity', 'Room_Volume', 'AvailableRoomVolume', 'Humidity', 'Air_Velocity', 
               'AC_Setpoint(MRT)', 'Desired_AirVelocity', 'PMV_Target',
              'Metabolic_Rate', 'Clothing_Insulation', 'Age_Group', 'Age', 
              'Hour']
    
    return final_df[columns]




def main():
    """
    Main function to run the simulation twice with different PMV ranges and stack results
    """
    def run_simulation_with_pmv_range(pmv_min: float, pmv_max: float) -> pd.DataFrame:
        global find_comfort_parameters
        
        # Store original function
        original_func = find_comfort_parameters
        
        # Create modified function with specified PMV range
        def modified_find_comfort_parameters(*args, **kwargs):
            kwargs['target_pmv_min'] = pmv_min
            kwargs['target_pmv_max'] = pmv_max
            return original_func(*args, **kwargs)
        
        # Replace global function
        find_comfort_parameters = modified_find_comfort_parameters
        
        # Run simulation
        num_days = 365*10
        num_samples = num_days * 24 * 4
        room_size = 5
        room_volume = 36 # Initial room volume 3x4x3 (meters-each side)
        
        start_date = datetime(2024, 3, 1)
        timestamps = [start_date + timedelta(minutes=15 * i) for i in range(num_samples)]
        
        print(f"\nRunning simulation for PMV range ±{abs(pmv_min)}...")
        results = simulate_thermal_comfort(timestamps, room_size, room_volume)
        
        # Create DataFrame
        rows = []
        for i in range(len(timestamps)):
            curr_occupants = results['occupancy'][i]
            for j in range(curr_occupants):
                rows.append({
                    'Timestamp': timestamps[i],
                    'Occupant_Number': j + 1,
                    'Total_Occupants': curr_occupants,
                    'Actual_Indoor_Temp': round(results['starting_temps'][i], 2),
                    'Estimate_Indoor_Temp_due_to_Outdoor': round(results['indoor_temps_no_ac'][i, j], 0),
                    'Outdoor_Temperature': round(results['outdoor_temp'][i], 0),
                    'Room_Volume': round(results['room_volumes'][i], 2),
                    'AvailableRoomVolume': round(results['available_volumes'][i], 2),
                    'AC_Capacity': results['ac_capacities'][i],
                    'Humidity': round(results['humidities'][i, j], 0),
                    'Air_Velocity': round(results['air_velocities'][i, j], 2),
                    'PMV': round(np.clip(results['pmv_values'][i, j], -3, 3), 2),
                    'Metabolic_Rate': round(results['met_rates'][i, j], 2),
                    'Clothing_Insulation': round(results['clothing'][i], 2),
                    'Age_Group': results['age_groups'][i, j],
                    'Age': round(results['ages'][i, j], 0),
                    'Hour': timestamps[i].hour,
                    'Season': get_season(timestamps[i])
                })
        
        df = pd.DataFrame(rows)
        

        # Drop rows with PMV_
        
        # Process with elderly PMV
        df_processed = select_elderly_pmv(df)

        
        return df_processed
    
    # Run for PMV ±0.5
    df_05 = run_simulation_with_pmv_range(-0.5, 0.5)
    
    # Run for PMV ±1.0
    df_10 = run_simulation_with_pmv_range(-1.0, 1.0)
    
    # Combine dataframes vertically
    df_combined = pd.concat([df_05, df_10], axis=0, ignore_index=True)

    # Delete the rows where no PMV value could be found
    df_combined.dropna(subset=['PMV_Target'], inplace=True)

    # delete the categorical age group column
    df_combined.drop(columns=['Age_Group'], inplace=True)
    
    # Save combined dataset
    output_file = 'synthetic_ac_fan_data.csv'
    df_combined.to_csv(output_file, index=False)
    print(f"\nCombined dataset saved to '{output_file}'")
    
    # Print some summary statistics
    print("\nSummary of combined dataset:")
    print(f"Total samples: {len(df_combined)}")
    print(f"Total size of the dataset: {df_combined.shape}")


if __name__ == "__main__":
    # print the starting time like: "Start time: 2021-09-01 12:00:00"
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Start time: {start_time}")
    main()
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"End time: {end_time}")
    print("Time taken: ", datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S') - datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'))