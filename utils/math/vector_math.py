import math
from typing import Tuple, List

from utils import LoggerUtility


logger = LoggerUtility().get_logger()


# Calculating with a stationary ship assumption
def _calc_intercept_angle(
    ship_position: Tuple[float, float],
    bullet_speed: float,
    asteroid_position: Tuple[float, float],
    asteroid_velocity: Tuple[float, float],
) -> float:
    """
    Calculate the angle for the ship to shoot in order to intercept a moving asteroid.

    Args:
        ship_position (Tuple[float, float]): The (x, y) position of the ship.
        bullet_speed (float): The speed of the ship's bullets.
        asteroid_position (Tuple[float, float]): The (x, y) position of the asteroid.
        asteroid_velocity (Tuple[float, float]): The (x, y) velocity vector of the asteroid.

    Returns:
        float: The angle in degrees (0 to 360) that the ship needs to shoot to intercept the asteroid.
               Returns 0 if no valid intercept is possible.
    """
    # Cache math functions locally for faster lookups.
    _sqrt = math.sqrt
    _atan2 = math.atan2
    _degrees = math.degrees

    # Compute the relative position of the asteroid to the ship.
    dx = asteroid_position[0] - ship_position[0]
    dy = asteroid_position[1] - ship_position[1]

    asteroid_v_x, asteroid_v_y = asteroid_velocity

    # Compute quadratic coefficients: a*t^2 + b*t + c = 0.
    a = asteroid_v_x * asteroid_v_x + asteroid_v_y * asteroid_v_y - bullet_speed * bullet_speed
    b = 2 * (dx * asteroid_v_x + dy * asteroid_v_y)
    c = dx * dx + dy * dy
    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return 0

    sqrt_disc = _sqrt(discriminant)
    denominator = 2 * a

    t1 = (-b + sqrt_disc) / denominator
    t2 = (-b - sqrt_disc) / denominator

    # Determine the earliest valid (non-negative) time without using a list comprehension.
    if t1 >= 0 and (t2 < 0 or t1 <= t2):
        t_min = t1
    elif t2 >= 0:
        t_min = t2
    else:
        return 0

    # Calculate the intercept position using the earliest valid time.
    intercept_dx = dx + asteroid_v_x * t_min
    intercept_dy = dy + asteroid_v_y * t_min

    # Compute and normalize the intercept angle to the range [0, 360).
    intercept_angle = _degrees(_atan2(intercept_dy, intercept_dx)) % 360

    return intercept_angle


def heading_relative_angle(
    ship_position: Tuple[float, float],
    ship_heading: float,
    asteroid_position: Tuple[float, float],
) -> float:
    """
    Calculate the relative angle between the ship's heading and the direction to the asteroid.

    Args:
        ship_position (Tuple[float, float]): The (x, y) position of the ship.
        ship_heading (float): The ship's current heading in degrees.
        asteroid_position (Tuple[float, float]): The (x, y) position of the asteroid.

    Returns:
        float: The relative angle in degrees (0 to 360).
    """
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
):
    """
    Calculate the angle the ship needs to turn to intercept a moving asteroid.

    Args:
        ship_position (Tuple[float, float]): The (x, y) position of the ship.
        ship_heading (float): The current heading of the ship in degrees.
        ship_turn_rate_range (Tuple[float, float]): The maximum turn rates (left, right) in degrees per second.
        bullet_speed (float): The speed of the ship's bullets.
        asteroid_position (Tuple[float, float]): The (x, y) position of the asteroid.
        asteroid_velocity (Tuple[float, float]): The (x, y) velocity vector of the asteroid.
        delta_time (float): The time interval for the calculation.

    Returns:
        Tuple[float, bool]: A tuple containing the turn rate (positive for left, negative for right)
                            and a boolean indicating if the asteroid is nearly aligned.
    """
    # Compute the intercept angle and derive the angular difference.
    intercept_angle = _calc_intercept_angle(
        ship_position, bullet_speed, asteroid_position, asteroid_velocity
    )
    angle_delta = intercept_angle - ship_heading

    # If the angular difference is negligible, no turn is needed.
    if math.isclose(angle_delta, 0, abs_tol=1e-6):
        return 0, True

    # Unpack and adjust turn rate limits to avoid edge-case issues.
    left_turn_rate, right_turn_rate = ship_turn_rate_range
    left_turn_rate += 0.0001
    right_turn_rate -= 0.0001

    # Pre-calculate threshold values for efficiency.
    left_threshold = left_turn_rate * delta_time
    right_threshold = right_turn_rate * delta_time

    # Determine the appropriate turn rate based on the sign and magnitude of angle_delta.
    if 0 < angle_delta < 180:
        if angle_delta < left_threshold:
            return left_turn_rate, False
        elif angle_delta < 1:
            return angle_delta / delta_time, True
        else:
            return angle_delta / delta_time, False
    else:
        if angle_delta > right_threshold:
            return right_turn_rate, False
        elif angle_delta > -1:
            return angle_delta / delta_time, True
        else:
            return angle_delta / delta_time, False





def heading_and_speed_to_velocity(heading: float, speed: float) -> Tuple[float, float]:
    """
    Convert a heading (in degrees) and speed to x, y velocity components.
    
    Args:
        heading (float): The heading in degrees.
        speed (float): The speed magnitude.
    
    Returns:
        Tuple[float, float]: The (x, y) velocity components.
    """
    # Cache conversion to radians for speed.
    rad = math.radians(heading)
    return speed * math.cos(rad), speed * math.sin(rad)


def calculate_closure_rate(
    ship_position: Tuple[float, float],
    ship_heading: float,
    ship_speed: float,
    asteroid_position: Tuple[float, float],
    asteroid_velocity: Tuple[float, float],
) -> float:
    """
    Calculate the closure rate (rate at which the distance between the ship
    and an asteroid is decreasing).

    Args:
        ship_position (Tuple[float, float]): The (x, y) position of the ship.
        ship_heading (float): The heading of the ship in degrees.
        ship_speed (float): The speed of the ship.
        asteroid_position (Tuple[float, float]): The (x, y) position of the asteroid.
        asteroid_velocity (Tuple[float, float]): The (x, y) velocity of the asteroid.

    Returns:
        float: The closure rate (positive means the ship is closing in on the asteroid).
    """
    # Cache math.sqrt for faster repeated use.
    _sqrt = math.sqrt

    # Compute positional differences.
    dx = asteroid_position[0] - ship_position[0]
    dy = asteroid_position[1] - ship_position[1]

    # Get asteroid velocity components.
    asteroid_v_x, asteroid_v_y = asteroid_velocity

    # Compute ship velocity components.
    ship_v_x, ship_v_y = heading_and_speed_to_velocity(ship_heading, ship_speed)

    # Adjust values to avoid zero comparisons (preserving functionality).
    if ship_heading == 0:
        ship_heading = 1e-6
    if ship_heading == 180:
        ship_heading = 179.9999
    if asteroid_v_x == 0:
        asteroid_v_x = 1e-6
    if asteroid_v_y == 0:
        asteroid_v_y = 1e-6

    # Compute the Euclidean distance (with a small epsilon added to prevent division by zero).
    distance = _sqrt(dx * dx + dy * dy)
    # Calculate closure rate as the negative dot product of the relative position and relative velocity,
    # normalized by the distance.
    closure_rate = -((dx * (asteroid_v_x - ship_v_x) + dy * (asteroid_v_y - ship_v_y))
                     / (1e-6 + distance))
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
    """
    Calculate if the ship will collide with an asteroid and the time of collision.

    Args:
        ship_position (Tuple[float, float]): The (x, y) position of the ship.
        ship_heading (float): The heading of the ship in degrees.
        ship_speed (float): The speed of the ship.
        ship_radius (float): The radius of the ship.
        asteroid_position (Tuple[float, float]): The (x, y) position of the asteroid.
        asteroid_velocity (Tuple[float, float]): The (x, y) velocity vector of the asteroid.
        asteroid_radius (float): The radius of the asteroid.

    Returns:
        Tuple[bool, float]: A tuple where the first element indicates collision (True/False)
                            and the second element is the time until collision (or -1 if none).
    """
    # Cache frequently used math functions.
    _sqrt = math.sqrt
    _cos = math.cos
    _sin = math.sin
    _radians = math.radians

    # Convert ship heading to radians.
    ship_heading_rad = _radians(ship_heading)

    # Unpack positions.
    ship_x, ship_y = ship_position
    asteroid_x, asteroid_y = asteroid_position
    asteroid_v_x, asteroid_v_y = asteroid_velocity

    # Compute differences in position.
    dx = asteroid_x - ship_x
    dy = asteroid_y - ship_y

    # Compute relative velocity components.
    dv_x = asteroid_v_x - ship_speed * _cos(ship_heading_rad)
    dv_y = asteroid_v_y - ship_speed * _sin(ship_heading_rad)

    # Sum of radii determines collision threshold.
    R = ship_radius + asteroid_radius

    # Coefficients for the quadratic equation a*t^2 + b*t + c = 0.
    a = dv_x * dv_x + dv_y * dv_y
    b = 2 * (dx * dv_x + dy * dv_y)
    c = dx * dx + dy * dy - R * R

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False, -1

    sqrt_disc = _sqrt(discriminant)
    denominator = 2 * a
    t1 = (-b + sqrt_disc) / denominator
    t2 = (-b - sqrt_disc) / denominator

    # Determine the smallest non-negative collision time without using extra list allocations.
    t_min = 1e12  # Large initial value.
    if t1 >= 0 and t1 < t_min:
        t_min = t1
    if t2 >= 0 and t2 < t_min:
        t_min = t2
    if t_min == 1e12:
        return False, -1

    return True, t_min


def game_to_ship_frame(
    position_vector: Tuple[float, float],
    asteroid_positions: List[Tuple[float, float]],
    game_size: Tuple[float, float],
) -> Tuple[Tuple[float, float], ...]:
    """
    Convert asteroid positions from game coordinates to positions relative to the ship,
    accounting for map wrapping.

    Args:
        position_vector (Tuple[float, float]): The (x, y) position of the ship.
        asteroid_positions (List[Tuple[float, float]]): List of asteroid positions.
        game_size (Tuple[float, float]): The (width, height) dimensions of the game map.

    Returns:
        Tuple[Tuple[float, float], ...]: A tuple of relative positions (dx, dy) for each asteroid.
    """
    map_x, map_y = game_size
    old_x, old_y = position_vector

    relative_positions = []
    for ast in asteroid_positions:
        dx = ast[0] - old_x
        dy = ast[1] - old_y
        # Adjust for horizontal wrapping.
        if abs(dx) > map_x / 2:
            dx -= math.copysign(map_x, dx)
        # Adjust for vertical wrapping.
        if abs(dy) > map_y / 2:
            dy -= math.copysign(map_y, dy)
        relative_positions.append((dx, dy))
    return tuple(relative_positions)


def distance_to(relative_position: Tuple[float, float]) -> float:
    """
    Compute the Euclidean distance from a relative (dx, dy) position.

    Args:
        relative_position (Tuple[float, float]): The (dx, dy) relative position.

    Returns:
        float: The Euclidean distance.
    """
    dx, dy = relative_position
    return math.hypot(dx, dy)


def sort_by_distance(asteroid_positions: List[Tuple[float, float]]) -> List[int]:
    """
    Sort asteroid indices by their distance from the origin (0, 0).

    Args:
        asteroid_positions (List[Tuple[float, float]]): List of asteroid positions.

    Returns:
        List[int]: Sorted indices of asteroids by increasing distance.
    """
    # Cache math.sqrt locally.
    _sqrt = math.sqrt
    # Precompute distances using list comprehension.
    distances = [_sqrt(pos[0] * pos[0] + pos[1] * pos[1]) for pos in asteroid_positions]
    # Return sorted indices based on computed distances.
    return sorted(range(len(distances)), key=lambda k: distances[k])


def largest_gap_center(a):
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

    angle_delta = intercept_angle - ship_heading

    # If the angular difference is negligible, no turn is needed.
    if math.isclose(angle_delta, 0, abs_tol=1e-6):
        return 0, True

    # Unpack and adjust turn rate limits to avoid edge-case issues.
    left_turn_rate, right_turn_rate = ship_turn_rate_range
    left_turn_rate += 0.0001
    right_turn_rate -= 0.0001

    # Pre-calculate threshold values for efficiency.
    left_threshold = left_turn_rate * delta_time
    right_threshold = right_turn_rate * delta_time

    aim_tolerance = 0.3

    # Determine the appropriate turn rate based on the sign and magnitude of angle_delta.
    if 0 < angle_delta < 180:
        if angle_delta < left_threshold:
            return left_turn_rate, False
        elif angle_delta < aim_tolerance:
            return angle_delta / delta_time, True
        else:
            return angle_delta / delta_time, False
    else:
        if angle_delta > right_threshold:
            return right_turn_rate, False
        elif angle_delta > -aim_tolerance:
            return angle_delta / delta_time, True
        else:
            return angle_delta / delta_time, False