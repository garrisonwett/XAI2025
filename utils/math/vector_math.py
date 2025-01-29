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


def heading_relative_angle(ship_position, ship_heading, asteroid_position):

    dx, dy = asteroid_position[0] - ship_position[0], asteroid_position[1] - ship_position[1]
    angle = math.degrees(math.atan2(dy, dx)) % 360

    relative_angle = (angle - ship_heading) % 360

    return relative_angle

def turn_angle(
    ship_position: Tuple[float, float],
    ship_heading: float,
    ship_turn_rate_range: Tuple[float, float],
    bullet_speed: float,
    asteroid_position: Tuple[float, float],
    asteroid_velocity: Tuple[float, float],
    delta_time: float,
) :
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
    
    # If the asteroid is already in the direction of the ship, no turn is needed
    if math.isclose(angle_delta, 0, abs_tol=1e-6):
        return 0, True
    left_turn_rate, right_turn_rate = ship_turn_rate_range
    left_turn_rate = left_turn_rate + 0.0001
    right_turn_rate = right_turn_rate - 0.0001
    # Determine the appropriate turn rate
    if 0 < angle_delta < 180:
        if angle_delta < left_turn_rate * delta_time:
            return left_turn_rate, False
        elif angle_delta < 1:
            return angle_delta / delta_time, True
        else:
            return angle_delta / delta_time, False
    else:
        if angle_delta > right_turn_rate * delta_time:
            return right_turn_rate, False
        elif angle_delta > -1:
            return angle_delta / delta_time, True
        else:
            return angle_delta / delta_time, False

def heading_and_speed_to_velocity(heading: float, speed: float) -> Tuple[float, float]:

    heading_rad = math.radians(heading)
    return speed * math.cos(heading_rad), speed * math.sin(heading_rad)


def calculate_closure_rate(
    ship_position: Tuple[float, float],
    ship_heading: float,
    ship_speed: float,
    asteroid_position: Tuple[float, float],
    asteroid_velocity: Tuple[float, float],
) -> float:
    

    dx, dy = asteroid_position[0] - ship_position[0], asteroid_position[1] - ship_position[1]
    asteroid_v_x, asteroid_v_y = asteroid_velocity
    ship_v_x, ship_v_y = heading_and_speed_to_velocity(ship_heading, ship_speed)

    closure_rate = -1 * (dx * (asteroid_v_x - ship_v_x) + dy * (asteroid_v_y - ship_v_y)) / math.sqrt(dx**2 + dy**2)
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
        ship_position (`Tuple[float, float]`): The (x, y) position of the ship.
        ship_heading (`float`): The heading of the ship in degrees.
        ship_speed (`float`): The speed of the ship.
        ship_radius (`float`): The radius of the ship.
        asteroid_position (`Tuple[float, float]`): The (x, y) position of the asteroid.
        asteroid_velocity (`Tuple[float, float]`): The (x, y) velocity vector of the asteroid.
        asteroid_radius (`float`): The radius of the asteroid.

    Returns:
        `Tuple[bool, float]`: A tuple of collision boolean and the time of collision
    """

    ship_heading_rad = math.radians(ship_heading)

    ship_x, ship_y = ship_position
    asteroid_x, asteroid_y = asteroid_position
    asteroid_v_x, asteroid_v_y = asteroid_velocity

    dx, dy = asteroid_x - ship_x, asteroid_y - ship_y
    dv_x, dv_y = asteroid_v_x - ship_speed * (
        math.cos(ship_heading_rad)
    ), asteroid_v_y - ship_speed * (math.sin(ship_heading_rad))

    R = ship_radius + asteroid_radius

    a = dv_x**2 + dv_y**2
    b = 2 * (dx * dv_x + dy * dv_y)
    c = dx**2 + dy**2 - R**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return False, -1

    sqrt_disc = math.sqrt(discriminant)

    denominator = 2 * a
    t1 = (-b + sqrt_disc) / denominator
    t2 = (-b - sqrt_disc) / denominator

    # Find the earliest valid collision time (non-negative)
    valid_collision_times = [t for t in [t1, t2] if t >= 0]
    if not valid_collision_times:
        return False, -1
    t_min = min(valid_collision_times)

    return True, t_min

def game_to_ship_frame(
    position_vector: list[float, float],
    asteroid_positions: Tuple[float, float],
    game_size: list[float, float],
) -> float:
    map_x, map_y = game_size
    old_x, old_y = position_vector
    
    relative_positions = []

    for asteroid in range(len(asteroid_positions)):
        ast_x, ast_y = asteroid_positions[asteroid]
        dx = ast_x - old_x
        dy = ast_y - old_y
        if abs(dx) > map_x / 2:
            dx -= map_x * dx / abs(dx)
        if abs(dy) > map_y / 2:
            dy -= map_y * dy / abs(dy)
        relative_positions.append((dx, dy))
        

    return tuple(relative_positions)
    
def distance_to(relative_position):
    dx, dy = relative_position
    return math.sqrt(dx**2 + dy**2)

