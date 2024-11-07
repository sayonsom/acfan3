from pythermalcomfort.models import pmv_ppd
from typing import Tuple
import numpy as np

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
    
    # First try to find solution with initial air velocity
    def binary_search_mrt(vr_current: float) -> Tuple[float, float, int]:
        left = mrt_min
        right = mrt_max
        iterations = 0
        max_iterations = 50
        
        while (right - left) > tolerance and iterations < max_iterations:
            mrt = (left + right) / 2
            current_pmv = calculate_pmv(mrt, vr_current)
            
            if target_pmv_min <= current_pmv <= target_pmv_max:
                return mrt, current_pmv, iterations
            elif current_pmv > target_pmv_max:
                right = mrt
            else:
                left = mrt
            iterations += 1
        
        final_mrt = (left + right) / 2
        final_pmv = calculate_pmv(final_mrt, vr_current)
        return final_mrt, final_pmv, iterations

    # Try initial solution with given air velocity
    mrt, pmv, iterations = binary_search_mrt(vr)
    
    # If MRT is below minimum acceptable, try adjusting air velocity
    if mrt < min_mrt:
        print(f"\nMRT {mrt:.2f}°C is below minimum {min_mrt}°C. Attempting to adjust air velocity...")
        
        left_vr = vr_min
        right_vr = vr_max
        vr_iterations = 0
        max_vr_iterations = 50
        
        while (right_vr - left_vr) > tolerance and vr_iterations < max_vr_iterations:
            current_vr = (left_vr + right_vr) / 2
            current_pmv = calculate_pmv(min_mrt, current_vr)
            
            if target_pmv_min <= current_pmv <= target_pmv_max:
                return min_mrt, current_vr, current_pmv, iterations + vr_iterations
            elif current_pmv > target_pmv_max:
                left_vr = current_vr  # increase air velocity to reduce PMV
            else:
                right_vr = current_vr  # decrease air velocity to increase PMV
            
            vr_iterations += 1
        
        # Return best found solution with adjusted air velocity
        final_vr = (left_vr + right_vr) / 2
        final_pmv = calculate_pmv(min_mrt, final_vr)
        return min_mrt, final_vr, final_pmv, iterations + vr_iterations
    
    return mrt, vr, pmv, iterations

def explore_conditions(base_conditions: dict, mrt: float, vr: float, range_size: int = 5):
    """Explore PMV values for variations in MRT and air velocity"""
    mrt_range = np.linspace(mrt - range_size, mrt + range_size, 5)
    vr_range = np.linspace(max(0.1, vr - 0.2), vr + 0.2, 5)
    
    print("\nPMV values for different conditions:")
    print("\nVarying MRT with fixed air velocity:")
    for test_mrt in mrt_range:
        result = pmv_ppd(tr=test_mrt, vr=vr, **{k: v for k, v in base_conditions.items() if k != 'vr'})
        print(f"MRT: {test_mrt:.1f}°C, vr: {vr:.2f} m/s → PMV: {result['pmv']:.3f}")
    
    print("\nVarying air velocity with fixed MRT:")
    for test_vr in vr_range:
        result = pmv_ppd(tr=mrt, vr=test_vr, **{k: v for k, v in base_conditions.items() if k != 'vr'})
        print(f"MRT: {mrt:.1f}°C, vr: {test_vr:.2f} m/s → PMV: {result['pmv']:.3f}")

# Example usage
if __name__ == "__main__":
    # Example parameters
    conditions = {
        'tdb': 29,    # Air temperature (°C)
        'rh': 50,     # Relative humidity (%)
        'vr': 0.1,    # Initial relative air velocity (m/s)
        'met': 1.2,   # Metabolic rate (met)
        'clo': 0.5,   # Clothing insulation (clo)
    }
    
    print("Initial conditions:")
    for key, value in conditions.items():
        print(f"  {key}: {value}")
    
    try:
        mrt, final_vr, pmv, iterations = find_comfort_parameters(**conditions)
        print(f"\nSolution found after {iterations} iterations:")
        print(f"Found MRT: {mrt:.2f}°C")
        print(f"Required air velocity: {final_vr:.2f} m/s")
        print(f"Resulting PMV: {pmv:.3f}")
        
        # Explore conditions around the solution
        explore_conditions(conditions, mrt, final_vr)
            
    except ValueError as e:
        print(f"\nError: {e}")