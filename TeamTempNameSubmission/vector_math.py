import math
from typing import Tuple, List
import numpy as np


# =============================================================================
# HIGH-LEVEL CONTROL FUNCTIONS
# =============================================================================

def turn_angle(
    ship_position: Tuple[float, float],
    ship_velocity: Tuple[float, float],
    ship_heading: float,
    ship_turn_rate_range: Tuple[float, float],
    bullet_speed: float,
    asteroid_position: Tuple[float, float],
    asteroid_velocity: Tuple[float, float],
    delta_time: float,
    extra_tolerance: float = -0.5
):
    """
    Calculate turn rate to aim at the intercept point of an asteroid.

    ship_turn_rate_range is (min_rate, max_rate), e.g. (-180, 180):
      min_rate = most-negative (fastest clockwise)
      max_rate = most-positive (fastest counter-clockwise)

    angle_delta > 0 → target is CCW from heading → positive turn rate needed
    angle_delta < 0 → target is CW  from heading → negative turn rate needed
    """
    # 1. Intercept angle accounting for ship velocity
    intercept_angle = _calc_intercept_angle(
        ship_position, ship_velocity, bullet_speed,
        asteroid_position, asteroid_velocity
    )

    # 2. Signed error in (-180, 180]
    angle_delta = normalize_angle(intercept_angle - ship_heading)

    # 3. Dynamic fire-tolerance based on asteroid angular size
    dist = max(math.hypot(
        asteroid_position[0] - ship_position[0],
        asteroid_position[1] - ship_position[1]
    ), 1.0)

    ratio = min(max(8.0 / dist, -1.0), 1.0)
    geometric_tolerance = math.degrees(math.asin(ratio))
    final_tolerance = geometric_tolerance + extra_tolerance

    is_aligned = abs(angle_delta) <= final_tolerance

    # 4. Dead-zone — close enough, stop turning and fire
    if abs(angle_delta) < 0.1:
        return 0.0, True

    # 5. Proportional turn with rate clamping
    min_rate, max_rate = ship_turn_rate_range        # e.g. (-180, +180)

    if angle_delta > 0:
        # Need CCW (positive) turn
        max_step = max_rate * delta_time             # max degrees we can cover this frame
        if angle_delta <= max_step:
            return angle_delta / delta_time, is_aligned
        else:
            return max_rate, is_aligned
    else:
        # Need CW (negative) turn
        max_step = abs(min_rate) * delta_time
        if abs(angle_delta) <= max_step:
            return angle_delta / delta_time, is_aligned   # angle_delta is already negative
        else:
            return min_rate, is_aligned


def go_to_angle(
    ship_heading: float,
    ship_turn_rate_range: Tuple[float, float],
    target_angle: float,
    delta_time: float,
):
    """Turn toward a target heading. Returns (turn_rate, arrived)."""
    angle_delta = normalize_angle(target_angle - ship_heading)

    if abs(angle_delta) < 1.0:
        return 0.0, True

    min_rate, max_rate = ship_turn_rate_range

    if angle_delta > 0:
        if angle_delta <= max_rate * delta_time:
            return angle_delta / delta_time, True
        return max_rate, False
    else:
        if abs(angle_delta) <= abs(min_rate) * delta_time:
            return angle_delta / delta_time, True
        return min_rate, False


# =============================================================================
# LOW-LEVEL VECTOR MATH & HELPERS
# =============================================================================

def normalize_angle(angle: float) -> float:
    """Normalize an angle to (-180, 180]."""
    return (angle + 180) % 360 - 180


def wrapped_delta(pos1, pos2, map_size):
    """Shortest (dx, dy) from pos1 to pos2, accounting for map wrapping."""
    width, height = map_size
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]

    if dx > width / 2:
        dx -= width
    elif dx < -width / 2:
        dx += width

    if dy > height / 2:
        dy -= height
    elif dy < -height / 2:
        dy += height

    return dx, dy


def _calc_intercept_angle(
    ship_position: Tuple[float, float],
    ship_velocity: Tuple[float, float],
    bullet_speed: float,
    asteroid_position: Tuple[float, float],
    asteroid_velocity: Tuple[float, float],
) -> float:
    """
    Angle the ship must face so a bullet (which inherits ship velocity)
    intercepts the asteroid.  Falls back to direct bearing if no solution.
    """
    dx = asteroid_position[0] - ship_position[0]
    dy = asteroid_position[1] - ship_position[1]

    # Relative velocity — asteroid motion in the bullet's rest frame
    rel_vx = asteroid_velocity[0] - ship_velocity[0]
    rel_vy = asteroid_velocity[1] - ship_velocity[1]

    # Quadratic: |D + V_rel*t|^2 = (bullet_speed * t)^2
    a = rel_vx**2 + rel_vy**2 - bullet_speed**2
    b = 2.0 * (dx * rel_vx + dy * rel_vy)
    c = dx**2 + dy**2

    discriminant = b * b - 4.0 * a * c

    # Fallback: aim directly at current position
    direct_angle = math.degrees(math.atan2(dy, dx)) % 360

    if discriminant < 0:
        return direct_angle

    sqrt_disc = math.sqrt(discriminant)

    if abs(a) < 1e-9:
        if abs(b) < 1e-9:
            return direct_angle
        t = -c / b
        t_min = t if t > 0 else None
    else:
        t1 = (-b + sqrt_disc) / (2.0 * a)
        t2 = (-b - sqrt_disc) / (2.0 * a)
        positives = [t for t in (t1, t2) if t > 0]
        t_min = min(positives) if positives else None

    if t_min is None:
        return direct_angle

    aim_dx = dx + rel_vx * t_min
    aim_dy = dy + rel_vy * t_min
    return math.degrees(math.atan2(aim_dy, aim_dx)) % 360


def solve_intercept_time(ship_pos, ship_vel, bullet_speed, target_pos, target_vel):
    """
    Time 't' when a bullet fired now would reach the target.
    Returns None if no positive-time solution exists.
    """
    dp = np.array(target_pos) - np.array(ship_pos)
    dv = np.array(target_vel) - np.array(ship_vel)

    A = np.dot(dv, dv) - bullet_speed**2
    B = 2.0 * np.dot(dp, dv)
    C = np.dot(dp, dp)

    discriminant = B * B - 4.0 * A * C
    if discriminant < 0:
        return None

    sqrt_disc = math.sqrt(discriminant)

    if abs(A) < 1e-8:
        if abs(B) < 1e-8:
            return None
        t = -C / B
        return t if t > 0 else None

    t1 = (-B - sqrt_disc) / (2.0 * A)
    t2 = (-B + sqrt_disc) / (2.0 * A)

    positives = [t for t in (t1, t2) if t > 0]
    return min(positives) if positives else None


def heading_relative_angle(
    ship_position: Tuple[float, float],
    ship_heading: float,
    asteroid_position: Tuple[float, float],
) -> float:
    """Angle from ship to asteroid relative to ship heading, in [0, 360)."""
    dx = asteroid_position[0] - ship_position[0]
    dy = asteroid_position[1] - ship_position[1]
    angle = math.degrees(math.atan2(dy, dx)) % 360
    return (angle - ship_heading) % 360


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
    """Positive = asteroid is approaching the ship."""
    dx = asteroid_position[0] - ship_position[0]
    dy = asteroid_position[1] - ship_position[1]
    ship_vx, ship_vy = heading_and_speed_to_velocity(ship_heading, ship_speed)

    distance = max(math.hypot(dx, dy), 1e-6)
    return -(
        (dx * (asteroid_velocity[0] - ship_vx) + dy * (asteroid_velocity[1] - ship_vy))
        / distance
    )


def calculate_if_collide(
    ship_position: Tuple[float, float],
    ship_heading: float,
    ship_speed: float,
    ship_radius: float,
    asteroid_position: Tuple[float, float],
    asteroid_velocity: Tuple[float, float],
    asteroid_radius: float,
) -> Tuple[bool, float]:
    """Returns (will_collide, time_to_collision).  time = -1 if no collision."""
    ship_heading_rad = math.radians(ship_heading)
    dx = asteroid_position[0] - ship_position[0]
    dy = asteroid_position[1] - ship_position[1]

    dv_x = asteroid_velocity[0] - ship_speed * math.cos(ship_heading_rad)
    dv_y = asteroid_velocity[1] - ship_speed * math.sin(ship_heading_rad)

    R = ship_radius + asteroid_radius
    a = dv_x * dv_x + dv_y * dv_y
    b = 2.0 * (dx * dv_x + dy * dv_y)
    c = dx * dx + dy * dy - R * R

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0 or abs(a) < 1e-9:
        return False, -1.0

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    t2 = (-b - sqrt_disc) / (2.0 * a)

    positives = [t for t in (t1, t2) if t >= 0]
    if not positives:
        return False, -1.0
    return True, min(positives)


def game_to_ship_frame(
    position_vector: Tuple[float, float],
    asteroid_positions: List[Tuple[float, float]],
    game_size: Tuple[float, float],
) -> Tuple[Tuple[float, float], ...]:
    """Convert asteroid positions to ship-relative coordinates with wrapping."""
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
    return math.hypot(relative_position[0], relative_position[1])


def sort_by_distance(asteroid_positions: List[Tuple[float, float]]) -> List[int]:
    distances = [math.hypot(pos[0], pos[1]) for pos in asteroid_positions]
    return sorted(range(len(distances)), key=lambda k: distances[k])