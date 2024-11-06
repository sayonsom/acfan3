from pythermalcomfort.models import pmv_ppd
from typing import Tuple
import numpy as np

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

# Example usage
if __name__ == "__main__":
    # Example parameters
    conditions = {
        'tdb': 29,    # Air temperature (°C)
        'rh': 50,     # Relative humidity (%)
        'vr': 0.1,    # Relative air velocity (m/s)
        'met': 1.2,   # Metabolic rate (met)
        'clo': 0.5,   # Clothing insulation (clo)
    }
    
    print("Initial conditions:")
    for key, value in conditions.items():
        print(f"  {key}: {value}")
    
    try:
        mrt, pmv, iterations = find_mrt_for_pmv_range(**conditions)
        print(f"\nSolution found after {iterations} iterations:")
        print(f"Found MRT: {mrt:.2f}°C")
        print(f"Resulting PMV: {pmv:.3f}")
        
        # Show PMV values for range of MRT temperatures around solution
        print("\nPMV values for different MRT temperatures:")
        mrt_range = np.linspace(mrt - 5, mrt + 5, 5)
        pmv_values = explore_mrt_range(conditions, mrt_range)
        for mrt_temp, pmv_val in zip(mrt_range, pmv_values):
            print(f"MRT: {mrt_temp:.1f}°C → PMV: {pmv_val:.3f}")
            
    except ValueError as e:
        print(f"\nError: {e}")