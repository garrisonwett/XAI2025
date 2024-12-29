from typing import Tuple

import numpy as np

from utils import LoggerUtility


logger = LoggerUtility().get_logger()


# Calculating with a stationary ship assumption
def _calc_intercept_angle(
    ship_position: Tuple[float, float],
    bullet_speed: float,
    asteroid_position: Tuple[float, float],
    asteroid_velocity: Tuple[float, float],
    delta_time: float,
) -> float:
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

    sqrt_disc = np.sqrt(discriminant)

    t1 = (-b + sqrt_disc) / (2 * a) + delta_time
    t2 = (-b - sqrt_disc) / (2 * a) + delta_time
    t_min = min([t for t in [t1, t2] if t >= 0])
    intercept_dx = dx + asteroid_v_x * t_min
    intercept_dy = dy + asteroid_v_y * t_min

    intercept_angle = np.arctan2(intercept_dy, intercept_dx) * 180 / np.pi
    intercept_angle = intercept_angle % 360

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
    angle_delta = (
        _calc_intercept_angle(
            ship_position,
            bullet_speed,
            asteroid_position,
            asteroid_velocity,
            delta_time,
        )
        - ship_heading
    )
    logger.debug(f"Angle delta: {angle_delta}")

    # The asteroid is already in the direction of the ship
    if angle_delta == 0:
        return 0

    left_turn_rate, right_turn_rate = ship_turn_rate_range
    if angle_delta < 180:
        if angle_delta < left_turn_rate * delta_time:
            return left_turn_rate
        else:
            return angle_delta - (angle_delta / 6)
    else:
        if angle_delta > right_turn_rate * delta_time:
            return right_turn_rate
        else:
            return angle_delta + (angle_delta / 6)
