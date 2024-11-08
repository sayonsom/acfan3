import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_air_speed(blade_diameter, rpm, cmm, distance_below=2.0, num_blades=3):
    """
    Calculate air speed using logarithmic and exponential relationships
    
    Parameters:
    blade_diameter (float): Diameter of the fan in meters
    rpm (float): Rotations per minute
    cmm (float): Air flow rate in cubic meters per minute
    num_blades (int): Number of blades (3 or 4)
    
    Returns:
    float: Air speed in meters per second
    """
    # Convert CMM to m³/s
    flow_rate = cmm / 60
    
    # Calculate cross-sectional area
    radius = blade_diameter / 2
    area = math.pi * radius**2
    
    # Diameter-based velocity calculation using logarithmic decay
    diameter_mm = blade_diameter * 1000
    
    # Logarithmic decay for diameter
    # Using log function to create natural decay from 4 m/s to 1 m/s
    log_factor = math.log(diameter_mm/600) / math.log(1600/600)
    velocity = 4.0 * math.exp(-1.5 * log_factor)
    
    # Exponential relationship with CMM
    # Reference CMM is 225
    cmm_factor = math.pow(cmm/225, 0.5)  # Square root relationship
    velocity *= cmm_factor
    
    # Linear relationship with RPM
    # Reference RPM is 310
    velocity *= (rpm / 310)
    
    # Adjust for number of blades
    if num_blades == 4:
        velocity *= 0.92
    
    # Apply distance decay (2.0m)
    distance_decay = 0.15  # 15% reduction at 2m
    velocity *= (1 - distance_decay)
    
    return max(velocity, 0)


def validate_parameters(diameter_mm, rpm, cmm):
    """Validate input parameters are within acceptable ranges"""
    if not (600 <= diameter_mm <= 1600):
        raise ValueError("Diameter must be between 600mm and 1600mm")
    if not (250 <= rpm <= 400):
        raise ValueError("RPM must be between 250 and 400")
    if not (150 <= cmm <= 300):
        raise ValueError("CMM must be between 150 and 300")

def create_analysis_plots(diameter_mm, rpm, cmm):
    diameter_m = diameter_mm / 1000
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Diameter vs Air Speed (fixed RPM and CMM)
    diameters = np.linspace(600, 1600, 100) / 1000  # More points for smoother curve
    speeds_3_d = [calculate_air_speed(d, rpm, cmm, num_blades=3) for d in diameters]
    speeds_4_d = [calculate_air_speed(d, rpm, cmm, num_blades=4) for d in diameters]
    
    ax1 = fig.add_subplot(221)
    ax1.plot(diameters * 1000, speeds_3_d, 'b-', label='3 Blades', linewidth=2)
    ax1.plot(diameters * 1000, speeds_4_d, 'r--', label='4 Blades', linewidth=2)
    ax1.set_xlabel('Diameter (mm)')
    ax1.set_ylabel('Air Speed (m/s)')
    ax1.set_title(f'Air Speed vs Diameter\n(RPM={rpm}, CMM={cmm})')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim(0, 4.5)
    
    # 2. RPM vs Air Speed (fixed diameter and CMM)
    rpms = np.linspace(250, 400, 50)
    speeds_3_r = [calculate_air_speed(diameter_m, r, cmm, num_blades=3) for r in rpms]
    speeds_4_r = [calculate_air_speed(diameter_m, r, cmm, num_blades=4) for r in rpms]
    
    ax2 = fig.add_subplot(222)
    ax2.plot(rpms, speeds_3_r, 'b-', label='3 Blades')
    ax2.plot(rpms, speeds_4_r, 'r--', label='4 Blades')
    ax2.set_xlabel('RPM')
    ax2.set_ylabel('Air Speed (m/s)')
    ax2.set_title(f'Air Speed vs RPM\n(Diameter={diameter_mm}mm, CMM={cmm})')
    ax2.grid(True)
    ax2.legend()
    ax2.set_ylim(0, 4.5)
    
    # 3. CMM vs Air Speed (fixed diameter and RPM)
    cmms = np.linspace(150, 300, 50)
    speeds_3_c = [calculate_air_speed(diameter_m, rpm, c, num_blades=3) for c in cmms]
    speeds_4_c = [calculate_air_speed(diameter_m, rpm, c, num_blades=4) for c in cmms]
    
    ax3 = fig.add_subplot(223)
    ax3.plot(cmms, speeds_3_c, 'b-', label='3 Blades')
    ax3.plot(cmms, speeds_4_c, 'r--', label='4 Blades')
    ax3.set_xlabel('CMM (m³/min)')
    ax3.set_ylabel('Air Speed (m/s)')
    ax3.set_title(f'Air Speed vs CMM\n(Diameter={diameter_mm}mm, RPM={rpm})')
    ax3.grid(True)
    ax3.legend()
    ax3.set_ylim(0, 4.5)
    
    # 4. Comparative bar plot for current configuration
    ax4 = fig.add_subplot(224)
    current_speed_3 = calculate_air_speed(diameter_m, rpm, cmm, num_blades=3)
    current_speed_4 = calculate_air_speed(diameter_m, rpm, cmm, num_blades=4)
    
    bars = ax4.bar(['3 Blades', '(Hypothetical) 4 Blades'], 
                   [current_speed_3, current_speed_4],
                   color=['blue', 'red'])
    ax4.set_ylabel('Air Speed (m/s)')
    ax4.set_title('Polycab Configuation\n(2.0m below fan)')
    ax4.set_ylim(0, 4.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} m/s',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def analyze_fan_configuration(diameter_mm, rpm, cmm):
    """Analyze fan configuration and display results"""
    try:
        validate_parameters(diameter_mm, rpm, cmm)
        
        # Convert diameter to meters for calculations
        diameter_m = diameter_mm / 1000
        
        print("\nFan Configuration Analysis")
        print("-" * 40)
        print(f"Diameter: {diameter_mm} mm")
        print(f"RPM: {rpm}")
        print(f"CMM: {cmm} m³/min")
        print("\nAir Speed at 2.0m below fan:")
        print("-" * 40)
        
        speed_3 = calculate_air_speed(diameter_m, rpm, cmm, num_blades=3)
        speed_4 = calculate_air_speed(diameter_m, rpm, cmm, num_blades=4)
        
        print(f"3-Blade Configuration: {speed_3:.2f} m/s")
        print(f"4-Blade Configuration: {speed_4:.2f} m/s")
        print(f"Difference: {speed_4 - speed_3:+.2f} m/s")
        
        # Create and display plots
        fig = create_analysis_plots(diameter_mm, rpm, cmm)
        return fig
        
    except ValueError as e:
        print(f"Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Example configuration
    diameter_mm = 1200  # 1200mm
    rpm = 310
    cmm = 225
    
    # Run analysis and create plots
    fig = analyze_fan_configuration(diameter_mm, rpm, cmm)
    if fig:
        plt.show()