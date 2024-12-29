import math
from typing import Tuple

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
        ship_position (`Tuple[float, float]`): The (x, y) position of the ship.
        bullet_speed (`float`): The speed of the ship's bullets.
        asteroid_position (`Tuple[float, float]`): The (x, y) position of the asteroid.
        asteroid_velocity (`Tuple[float, float]`): The (x, y) velocity vector of the asteroid.

    Returns:
        `float`: The angle in degrees (0 to 360) that the ship needs to shoot to intercept the asteroid.
               Returns 0 if no valid intercept is possible.
    """    
    # Calculate the relative position of the asteroid to the ship
    dx, dy = (
        asteroid_position[0] - ship_position[0],
        asteroid_position[1] - ship_position[1],
    )

    asteroid_v_x, asteroid_v_y = asteroid_velocity

    # Quadratic formula to solve for the time of intercept
    a = asteroid_v_x**2 + asteroid_v_y**2 - bullet_speed**2
    b = 2 * (dx * asteroid_v_x + dy * asteroid_v_y)
    c = dx**2 + dy**2
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return 0

    # Solve the quadratic equation for time
    sqrt_disc = math.sqrt(discriminant)
    denominator = 2 * a

    t1 = (-b + sqrt_disc) / denominator
    t2 = (-b - sqrt_disc) / denominator

    # Find the earliest valid time (non-negative)
    valid_times = [t for t in [t1, t2] if t >= 0]
    if not valid_times:
        return 0
    t_min = min(valid_times)

    # Calculate the intercept position
    intercept_dx = dx + asteroid_v_x * t_min
    intercept_dy = dy + asteroid_v_y * t_min

    # Calculate the angle to intercept and normalize to 0-360 degrees
    intercept_angle = math.degrees(math.atan2(intercept_dy, intercept_dx)) % 360

    return intercept_angle

def turn_angle(
    ship_position: Tuple[float, float],
    ship_heading: float,
    ship_turn_rate_range: Tuple[float, float],
    bullet_speed: float,
    asteroid_position: Tuple[float, float],
    asteroid_velocity: Tuple[float, float],
    delta_time: float,
) -> float:
    """
    Calculate the angle the ship needs to turn to intercept a moving asteroid.

    Args:
        ship_position (`Tuple[float, float]`): The (x, y) position of the ship.
        ship_heading (`float`): The current heading of the ship in degrees.
        ship_turn_rate_range (`Tuple[float, float]`): The maximum turn rates (left, right) in degrees per second.
        bullet_speed (`float`): The speed of the ship's bullets.
        asteroid_position (`Tuple[float, float]`): The (x, y) position of the asteroid.
        asteroid_velocity (`Tuple[float, float]`): The (x, y) velocity vector of the asteroid.
        delta_time (`float`): The time interval for the calculation.

    Returns:
        `float`: The turn rate (positive for left, negative for right) to align the ship with the intercept angle.
    """
    angle_delta = (
        _calc_intercept_angle(
            ship_position,
            bullet_speed,
            asteroid_position,
            asteroid_velocity,
        )
        - ship_heading
    )
    logger.debug(f"Angle delta: {angle_delta}")

    # If the asteroid is already in the direction of the ship, no turn is needed
    if math.isclose(angle_delta, 0, abs_tol=1e-6):
        return 0

    left_turn_rate, right_turn_rate = ship_turn_rate_range
    # Determine the appropriate turn rate
    if 0 < angle_delta < 180:
        if angle_delta < left_turn_rate * delta_time:
            return left_turn_rate
        else:
            return angle_delta / delta_time
    else:
        if angle_delta > right_turn_rate * delta_time:
            return right_turn_rate
        else:
            return angle_delta / delta_time

def calculate_if_collide(ship_position,ship_heading,ship_speed,ship_radius,asteroid_position,asteroid_velocity,asteroid_radius) -> Tuple[bool, float]:

    ship_heading_rad = ship_heading * math.pi / 180
    ship_x, ship_y = ship_position
    asteroid_x, asteroid_y = asteroid_position
    asteroid_v_x, asteroid_v_y = asteroid_velocity
    dx, dy = asteroid_x - ship_x, asteroid_y - ship_y
    dv_x, dv_y = asteroid_v_x - ship_speed*(math.cos(ship_heading_rad)), asteroid_v_y - ship_speed*(math.sin(ship_heading_rad))
    R = ship_radius + asteroid_radius

    a = dv_x**2 + dv_y**2
    b = 2*(dx*dv_x + dy*dv_y)
    c = dx**2 + dy**2 - R**2

    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        return False, -1
    
    sqrt_disc = math.sqrt(discriminant)

    t1 = (-b + sqrt_disc) / (2*a)
    t2 = (-b - sqrt_disc) / (2*a)

    collision_times = []
    for t_candidate in (t1, t2):
        if t_candidate >= 0:
            collision_times.append(t_candidate)

    if not collision_times:
        return False, -1
    
    collision_time = min(collision_times)
    return True, collision_time
    