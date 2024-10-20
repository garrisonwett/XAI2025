import numpy as np


def closest_approach(v1, v2, p1, p2):
    """
    Calculate the closest approach and the time at which it occurs.
    
    Args:
        v1, v2: numpy arrays representing the velocity vectors of the two objects
        p1, p2: numpy arrays representing the initial position vectors of the two objects
    
    Returns:
        t_closest: Time at which the closest approach occurs
        d_closest: Distance at the closest approach
    """
    # Relative velocity and position
    v_rel = v1 - v2
    p_rel = p1 - p2
    
    # Time of closest approach
    t_closest = -np.dot(p_rel, v_rel) / np.dot(v_rel, v_rel)
    
    # Position at closest approach
    closest_point = p_rel + t_closest * v_rel
    
    # Distance at closest approach
    d_closest = np.linalg.norm(closest_point)
    
    return t_closest, d_closest

print(closest_approach(np.array([1, 0]), np.array([1, 2]), np.array([0, 7]), np.array([0, 0])))  
