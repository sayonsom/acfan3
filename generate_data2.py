import numpy as np
import pandas as pd
from pythermalcomfort.models import pmv_ppd
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import List, Dict, Tuple, Any

# Set random seed for reproducibility
np.random.seed(2024)


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

def generate_starting_temperature(num_samples: int) -> np.ndarray:
    """
    Generate starting indoor temperatures using log-normal distribution
    
    Parameters:
    - num_samples: Number of samples to generate
    
    Returns:
    - np.ndarray: Array of starting temperatures
    """
    mu = np.log(25.5)
    sigma = 0.08
    temps = np.random.lognormal(mu, sigma, num_samples)
    return np.clip(temps, 24, 30)

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
    Generate humidity based on season and time
    
    Parameters:
    - timestamps: List of datetime objects
    
    Returns:
    - np.ndarray: Array of humidity values
    """
    num_samples = len(timestamps)
    humidity = np.zeros(num_samples)
    
    humidity_ranges = {
        'summer': (30, 50),
        'monsoon': (70, 90),
        'post-monsoon': (50, 70),
        'winter': (40, 60)
    }
    
    for i, ts in enumerate(timestamps):
        season = get_season(ts)
        base_range = humidity_ranges[season]
        base_humidity = np.random.uniform(base_range[0], base_range[1])
        
        hour = ts.hour + ts.minute/60
        time_variation = -10 * np.sin((2 * np.pi / 24) * (hour - 14))
        
        humidity[i] = np.clip(base_humidity + time_variation, 30, 90)
    
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
        'summer': {'mean': 0.6, 'sigma': 0.2},
        'monsoon': {'mean': 0.5, 'sigma': 0.3},
        'post-monsoon': {'mean': 0.3, 'sigma': 0.2},
        'winter': {'mean': 0.1, 'sigma': 0.2}
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
            base_met = np.random.uniform(2.0, 2.5)
        elif time_block in ['morning', 'afternoon']:
            base_met = np.random.uniform(1.0, 1.6)
        elif time_block == 'evening':
            base_met = np.random.uniform(1.4, 2.0)
        else:
            base_met = np.random.uniform(0.7, 1.0)
        
        met_rates[i] = adjust_metabolic_rate_for_age(base_met, age_groups[i])
        met_rates[i] += np.random.normal(0, 0.1)
    
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
    
    water_vapor_per_person = 50  # g/hour
    room_air_mass = room_volume * air_density
    humidity_increase = (water_vapor_per_person * num_occupants) / room_air_mass * 10
    new_humidity = min(base_humidity + humidity_increase, 90)
    
    velocity_factor = 1 + (num_occupants - 1) * 0.1
    
    return new_temp, new_humidity, velocity_factor

def simulate_thermal_comfort(timestamps: List[datetime], room_size: int = 5, 
                           room_volume: float = 36) -> Dict[str, Any]:
    """
    Simulate thermal comfort for multiple occupants
    """
    num_samples = len(timestamps)
    
    # Generate base environmental conditions
    outdoor_temp, base_indoor_temp = generate_temperature_cycles(timestamps)
    base_humidity = generate_humidity(timestamps)
    base_air_velocity = generate_air_velocity(timestamps)
    
    # Generate new columns
    starting_temps = generate_starting_temperature(num_samples)
    ac_capacities = generate_ac_capacity(num_samples)
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
            # Adjust room volume based on AC capacity
            adjusted_volume = room_volume * (ac_capacities[i] / 1.5)
            
            # Calculate room effects
            adj_temp, adj_humidity, vel_factor = calculate_room_effects(
                starting_temps[i],
                base_humidity[i],
                curr_occupants,
                adjusted_volume
            )
            
            # Apply adjustments to occupied positions
            for j in range(curr_occupants):
                indoor_temps[i,j] = adj_temp
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
        'humidities': humidities,
        'air_velocities': air_velocities,
        'pmv_values': pmv_values,
        'ppd_values': ppd_values,
        'age_groups': age_groups,
        'ages': ages,
        'outdoor_temp': outdoor_temp,
        'starting_temps': starting_temps,
        'ac_capacities': ac_capacities,
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


# Assuming df is your original dataframe
def select_elderly_pmv(df):
    """
    Select PMV values based on oldest occupant at each timestamp
    """
    # First identify the oldest person for each timestamp
    oldest_per_timestamp = df.groupby('Timestamp').apply(
        lambda x: x.loc[x['Age'].idxmax()]
    ).reset_index(drop=True)
    
    # Create final dataframe with one row per timestamp
    final_df = oldest_per_timestamp[['Timestamp', 'Occupant_Number', 'Total_Occupants', 'AC_SensorTemp_at_ServiceStart',
                                        'Indoor_Temperature_no_AC', 'Outdoor_Temperature', 'AC_Capacity', 'Humidity',
                                        'Air_Velocity', 'PMV', 'Metabolic_Rate', 'Clothing_Insulation', 'Age_Group', 'Age', 'Hour', 'Season']]
    
    return final_df


# For a set PMV range, find the desired set of MRT values
def find_mrt_for_pmv_range(df):
    """
    Find the MRT values that result in PMV within the target range
    """
    # Define target PMV range
    target_pmv_min = -0.2
    target_pmv_max = 0.2
    
    # Initialize lists to store results
    mrt_values = []
    pmv_values = []
    
    for i, row in df.iterrows():
        tdb = row['AC_SensorTemp_at_ServiceStart']
        rh = row['Humidity']
        vr = row['Air_Velocity']
        met = row['Metabolic_Rate']
        clo = row['Clothing_Insulation']
        
        try:
            mrt, pmv, _ = find_mrt_for_pmv_range(
                tdb=tdb, rh=rh, vr=vr, met=met, clo=clo,
                target_pmv_min=target_pmv_min, target_pmv_max=target_pmv_max
            )
            mrt_values.append(mrt)
            pmv_values.append(pmv)
        except ValueError as e:
            mrt_values.append(np.nan)
            pmv_values.append(np.nan)
    
    df['IndoorTemp_Desired'] = mrt_values
    df['PMV_Achievable'] = pmv_values
    
    return df

def main():
    """
    Main function to run the simulation
    """
    # Simulation parameters
    num_days = 7
    num_samples = num_days * 24 * 4  # 15-minute intervals
    room_size = 5
    room_volume = 36  # 3x4x3m room
    
    # Generate timestamps
    start_date = datetime(2024, 3, 1)
    timestamps = [start_date + timedelta(minutes=15 * i) for i in range(num_samples)]
    
    # Run simulation
    print("Starting thermal comfort simulation...")
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
                'AC_SensorTemp_at_ServiceStart': round(results['starting_temps'][i], 2),
                'Indoor_Temperature_no_AC': round(results['indoor_temps'][i, j], 2),
                'Outdoor_Temperature': round(results['outdoor_temp'][i], 2),  
                'AC_Capacity': round(results['ac_capacities'][i], 2),
                'Humidity': round(results['humidities'][i, j], 2),
                'Air_Velocity': round(results['air_velocities'][i, j], 2),
                'PMV': round(np.clip(results['pmv_values'][i, j], -3, 3), 2),
                # 'PPD': round(results['ppd_values'][i, j], 2),  # Uncomment if needed
                'Metabolic_Rate': round(results['met_rates'][i, j], 2),
                'Clothing_Insulation': round(results['clothing'][i], 2),
                'Age_Group': results['age_groups'][i, j],
                'Age': results['ages'][i, j],
                'Hour': timestamps[i].hour,
                'Season': get_season(timestamps[i])
            })

    
    df = pd.DataFrame(rows)

    # Drop the rows where PMV and PPD are NaN
    df = df.dropna(subset=['PMV'])

    
    # Create analysis plots
    # create_analysis_plots(df)

    df_elderly = select_elderly_pmv(df)

    df_final = find_mrt_for_pmv_range(df_elderly)

    # Save dataset
    output_file = 'synthetic_ac_fan_data.csv'
    df_final.to_csv(output_file, index=False)
    print(f"\nDataset saved to '{output_file}'")
    
    # # Print summary statistics
    # print("\nSummary Statistics by AC Capacity:")
    # print(df.groupby('AC_Capacity')[['PMV', 'Indoor_Temperature_no_AC', 'PPD']].describe())
    
    # print("\nOccupancy Distribution by Time Period:")
    # print(df.groupby('Hour')['Total_Occupants'].value_counts(normalize=True).mul(100).round(1))

if __name__ == "__main__":
    main()

def find_comfort_parameters(
    tdb: float,  # air temperature, °C
    rh: float,   # relative humidity, %
    vr: float,   # initial relative air velocity, m/s
    met: float,  # metabolic rate, met
    clo: float,  # clothing insulation, clo
    min_mrt: float = 26.0,  # minimum acceptable MRT
    target_pmv_min: float = -0.4,
    target_pmv_max: float = 0.4,
    mrt_min: float = 16,
    mrt_max: float = 32,
    vr_min: float = 0.1,
    vr_max: float = 0.8,  # maximum comfortable air velocity
    tolerance: float = 0.01
) -> Tuple[float, float, float, int]:
    """
    Find the MRT and air velocity that results in a PMV within the target range.
    If MRT would be below min_mrt, adjusts air velocity instead.
    
    Returns:
        Tuple[float, float, float, int]: (MRT, air velocity, resulting PMV, iterations)
    """
    
    def calculate_pmv(mrt: float, air_velocity: float) -> float:
        results = pmv_ppd(tdb=tdb, tr=mrt, rh=rh, vr=air_velocity, met=met, clo=clo)
        return results['pmv']



def generate_synthetic_data_with_comfort_params(num_samples: int) -> pd.DataFrame:
    """Generates synthetic data with integrated comfort parameters"""
    data = []
    for _ in range(num_samples):
        # Generate basic data points
        timestamp = datetime.now()
        occupant_number = np.random.randint(1, 5)
        total_occupants = np.random.randint(1, 10)
        ac_sensor_temp = np.random.uniform(20, 25)
        indoor_temp_no_ac = np.random.uniform(25, 30)
        outdoor_temp = np.random.uniform(15, 35)
        ac_capacity = np.random.choice([1.5, 2.0, 2.5])
        humidity = np.random.uniform(30, 70)
        pmv_initial = np.random.uniform(-1, 1)
        metabolic_rate = np.random.uniform(1.0, 1.4)
        clothing_insulation = np.random.uniform(0.5, 1.0)
        age_group = np.random.choice(['child', 'adult', 'senior'])
        age = np.random.randint(18, 65)
        hour = np.random.randint(0, 24)
        season = np.random.choice(['winter', 'summer', 'monsoon'])
        
        # Get MRT (AC_Setpoint) and desired air velocity using the comfort parameters function
        mrt, air_velocity, pmv_result, _ = find_comfort_parameters(
            tdb=ac_sensor_temp,
            rh=humidity,
            vr=0.2,  # Starting air velocity, can be parameterized if needed
            met=metabolic_rate,
            clo=clothing_insulation,
            target_pmv_min=-0.2,
            target_pmv_max=0.2
        )
        
        # Collecting data row
        data.append([
            timestamp, occupant_number, total_occupants, ac_sensor_temp,
            indoor_temp_no_ac, outdoor_temp, ac_capacity, humidity,
            air_velocity, pmv_initial, mrt, air_velocity, pmv_result, 
            metabolic_rate, clothing_insulation, age_group, age, hour, season
        ])

    # Define column names
    columns = [
        'Timestamp', 'Occupant_Number', 'Total_Occupants', 'AC_SensorTemp_at_ServiceStart',
        'Indoor_Temperature_no_AC', 'Outdoor_Temperature', 'AC_Capacity', 'Humidity',
        'Air_Velocity', 'PMV_Initial', 'AC_Setpoint(MRT)', 'Desired_AirVelocity', 'PMV_Target',
        'Metabolic_Rate', 'Clothing_Insulation', 'Age_Group', 'Age', 'Hour', 'Season'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    return df
