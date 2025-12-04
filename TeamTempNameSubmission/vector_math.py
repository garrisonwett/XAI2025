import math
from typing import Tuple, List
import numpy as np

from utils import LoggerUtility

logger = LoggerUtility().get_logger()

def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to be within the range [-180, 180].
    """
    return (angle + 180) % 360 - 180

def _calc_intercept_angle(
    ship_position: Tuple[float, float],
    bullet_speed: float,
    asteroid_position: Tuple[float, float],
    asteroid_velocity: Tuple[float, float],
) -> float:
    
    _sqrt = math.sqrt
    _atan2 = math.atan2
    _degrees = math.degrees

    dx = asteroid_position[0] - ship_position[0]
    dy = asteroid_position[1] - ship_position[1]

    asteroid_v_x, asteroid_v_y = asteroid_velocity

    a = asteroid_v_x**2 + asteroid_v_y**2 - bullet_speed**2
    b = 2 * (dx * asteroid_v_x + dy * asteroid_v_y)
    c = dx**2 + dy**2
    
    discriminant = b*b - 4*a*c

    if discriminant < 0:
        return 0

    sqrt_disc = _sqrt(discriminant)
    
    if abs(2*a) < 1e-9:
        return 0
        
    t1 = (-b + sqrt_disc) / (2*a)
    t2 = (-b - sqrt_disc) / (2*a)

    t_min = float('inf')
    if t1 >= 0: t_min = min(t_min, t1)
    if t2 >= 0: t_min = min(t_min, t2)
    
    if t_min == float('inf'):
        return 0

    intercept_dx = dx + asteroid_v_x * t_min
    intercept_dy = dy + asteroid_v_y * t_min

    intercept_angle = _degrees(_atan2(intercept_dy, intercept_dx)) % 360

    return intercept_angle


def heading_relative_angle(
    ship_position: Tuple[float, float],
    ship_heading: float,
    asteroid_position: Tuple[float, float],
) -> float:
    _atan2 = math.atan2
    _degrees = math.degrees

    dx = asteroid_position[0] - ship_position[0]
    dy = asteroid_position[1] - ship_position[1]
    angle = _degrees(_atan2(dy, dx)) % 360

    return (angle - ship_heading) % 360


def turn_angle(
    ship_position: Tuple[float, float],
    ship_heading: float,
    ship_turn_rate_range: Tuple[float, float],
    bullet_speed: float,
    asteroid_position: Tuple[float, float],
    asteroid_velocity: Tuple[float, float],
    delta_time: float,
    extra_tolerance: float = -0.5  # <--- NEW VARIABLE
):
    """
    Calculate turn to intercept with adjustable tolerance.
    extra_tolerance: Degrees to add to the geometric hit cone. 
                     Positive = Looser aim. Negative = Tighter aim.
    """
    # 1. Calculate Intercept Angle
    intercept_angle = _calc_intercept_angle(
        ship_position, bullet_speed, asteroid_position, asteroid_velocity
    )
    
    # 2. Normalize Delta (-180 to 180)
    angle_delta = normalize_angle(intercept_angle - ship_heading)

    # 3. Dynamic Tolerance Logic
    dist = math.hypot(
        asteroid_position[0] - ship_position[0],
        asteroid_position[1] - ship_position[1]
    )
    
    if dist < 1.0: dist = 1.0

    # Geometric tolerance: Angle from center to edge of asteroid (assuming radius 8)
    ratio = 8.0 / dist
    ratio = max(-1.0, min(1.0, ratio))
    
    geometric_tolerance = math.degrees(math.asin(ratio))
    
    # Final tolerance = Geometry + User Adjustment
    final_tolerance = geometric_tolerance + extra_tolerance

    # Check if we are aimed "good enough" to shoot
    is_aligned = abs(angle_delta) <= final_tolerance

    # 4. Handle tiny adjustments
    if abs(angle_delta) < 0.1:
        return 0.0, True

    # 5. Calculate Turn
    left_turn_rate, right_turn_rate = ship_turn_rate_range
    
    if angle_delta > 0:
        # Turn RIGHT (Negative)
        if angle_delta < abs(right_turn_rate * delta_time):
             return -angle_delta / delta_time, is_aligned
        else:
             return right_turn_rate, is_aligned
    else:
        # Turn LEFT (Positive)
        if abs(angle_delta) < (left_turn_rate * delta_time):
             return abs(angle_delta) / delta_time, is_aligned
        else:
             return left_turn_rate, is_aligned


def heading_and_speed_to_velocity(heading: float, speed: float) -> Tuple[float, float]:
    rad = math.radians(heading)
    return speed * math.cos(rad), speed * math.sin(rad)


def calculate_closure_rate(
    ship_position: Tuple[float, float],
    ship_heading: float,
    ship_speed: float,
    asteroid_position: Tuple[float, float],
    asteroid_velocity: Tuple[float, float],
) -> float:
    _sqrt = math.sqrt
    dx = asteroid_position[0] - ship_position[0]
    dy = asteroid_position[1] - ship_position[1]
    asteroid_v_x, asteroid_v_y = asteroid_velocity
    ship_v_x, ship_v_y = heading_and_speed_to_velocity(ship_heading, ship_speed)

    distance = _sqrt(dx * dx + dy * dy)
    if distance < 1e-6: distance = 1e-6
    
    closure_rate = -((dx * (asteroid_v_x - ship_v_x) + dy * (asteroid_v_y - ship_v_y))
                     / distance)
    return closure_rate


def calculate_if_collide(
    ship_position: Tuple[float, float],
    ship_heading: float,
    ship_speed: float,
    ship_radius: float,
    asteroid_position: Tuple[float, float],
    asteroid_velocity: Tuple[float, float],
    asteroid_radius: float,
) -> Tuple[bool, float]:
    _sqrt = math.sqrt
    _cos = math.cos
    _sin = math.sin
    _radians = math.radians

    ship_heading_rad = _radians(ship_heading)
    ship_x, ship_y = ship_position
    asteroid_x, asteroid_y = asteroid_position
    asteroid_v_x, asteroid_v_y = asteroid_velocity

    dx = asteroid_x - ship_x
    dy = asteroid_y - ship_y

    dv_x = asteroid_v_x - ship_speed * _cos(ship_heading_rad)
    dv_y = asteroid_v_y - ship_speed * _sin(ship_heading_rad)

    R = ship_radius + asteroid_radius

    a = dv_x * dv_x + dv_y * dv_y
    b = 2 * (dx * dv_x + dy * dv_y)
    c = dx * dx + dy * dy - R * R

    discriminant = b * b - 4 * a * c
    if discriminant < 0: return False, -1

    sqrt_disc = _sqrt(discriminant)
    if abs(2*a) < 1e-9: return False, -1
    
    t1 = (-b + sqrt_disc) / (2*a)
    t2 = (-b - sqrt_disc) / (2*a)

    t_min = 1e12
    if t1 >= 0 and t1 < t_min: t_min = t1
    if t2 >= 0 and t2 < t_min: t_min = t2
    if t_min == 1e12: return False, -1

    return True, t_min


def game_to_ship_frame(
    position_vector: Tuple[float, float],
    asteroid_positions: List[Tuple[float, float]],
    game_size: Tuple[float, float],
) -> Tuple[Tuple[float, float], ...]:
    map_x, map_y = game_size
    old_x, old_y = position_vector
    relative_positions = []
    for ast in asteroid_positions:
        dx = ast[0] - old_x
        dy = ast[1] - old_y
        if abs(dx) > map_x / 2:
            dx -= math.copysign(map_x, dx)
        if abs(dy) > map_y / 2:
            dy -= math.copysign(map_y, dy)
        relative_positions.append((dx, dy))
    return tuple(relative_positions)


def distance_to(relative_position: Tuple[float, float]) -> float:
    dx, dy = relative_position
    return math.hypot(dx, dy)


def sort_by_distance(asteroid_positions: List[Tuple[float, float]]) -> List[int]:
    _sqrt = math.sqrt
    distances = [_sqrt(pos[0] * pos[0] + pos[1] * pos[1]) for pos in asteroid_positions]
    return sorted(range(len(distances)), key=lambda k: distances[k])


def largest_gap_center(a):
    if not a: return 0
    a = sorted(a)
    gaps = [(a[i+1] - a[i], a[i]) for i in range(len(a)-1)] + [(a[0] + 1 - a[-1], a[-1])]
    d, s = max(gaps)
    return (s + d/2) % 1


def go_to_angle(
    ship_heading: float,
    ship_turn_rate_range: Tuple[float, float],
    intercept_angle: float,
    delta_time: float,
):
    angle_delta = normalize_angle(intercept_angle - ship_heading)

    if abs(angle_delta) < 1.0:
        return 0, True

    left_turn_rate, right_turn_rate = ship_turn_rate_range

    if angle_delta > 0:
        if angle_delta < abs(right_turn_rate * delta_time):
             return -angle_delta / delta_time, True
        else:
             return right_turn_rate, False
    else:
        if abs(angle_delta) < (left_turn_rate * delta_time):
             return abs(angle_delta) / delta_time, True
        else:
             return left_turn_rate, False
        

def speed_to_thrust(current_speed: float, target_speed: float) -> float:
    thrust = min(max(30 * (target_speed - current_speed), -500), 500)
    return thrust


def compute_safe_point_controls(
    ship_pos,
    ship_heading,
    thrust_range,
    turn_rate_range,
    map_width,
    map_height,
    asteroid_positions,
    asteroid_velocities,
    asteroid_radii,
    ship_radius
):
    xs = np.linspace(0, map_width, 20)
    ys = np.linspace(0, map_height, 20)

    best_point = None
    best_threat = float("inf")

    for x in xs:
        for y in ys:
            p = np.array([x, y])
            threat = 0.0
            for apos, avel, arad in zip(asteroid_positions, asteroid_velocities, asteroid_radii):
                apos = np.array(apos)
                avel = np.array(avel)
                d = np.linalg.norm(p - apos)
                if d < 1e-6: d = 1e-6
                rel = avel
                proj = abs(np.dot(rel, (p - apos)) / d)
                buffer_dist = max(ship_radius + arad, 1.0)
                threat += (proj + 1.0) / (d - buffer_dist + 1.0)

            if threat < best_threat:
                best_threat = threat
                best_point = p

    target_vec = best_point - np.array(ship_pos)
    target_angle = math.atan2(target_vec[1], target_vec[0])
    target_angle_deg = math.degrees(target_angle) % 360

    angle_diff = normalize_angle(target_angle_deg - ship_heading)
    turn_min, turn_max = turn_rate_range

    if angle_diff > 0:
        turn_rate = turn_min
    else:
        turn_rate = turn_max
        
    if abs(angle_diff) < abs(turn_max / 30):
        turn_rate = -angle_diff * 30 if angle_diff > 0 else abs(angle_diff) * 30

    turn_rate = max(min(turn_rate, turn_max), turn_min)

    alignment = 1.0 - abs(math.radians(angle_diff)) / math.pi
    thrust_min, thrust_max = thrust_range
    thrust = thrust_min + alignment * (thrust_max - thrust_min)

    return thrust, turn_rate