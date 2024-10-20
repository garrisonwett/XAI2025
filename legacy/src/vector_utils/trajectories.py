import numpy as np


def game_to_ship_frame(
    position_vector: list[float, float],
    other_position: list[float, float],
) -> float:
    x_diff = position_vector[0] - other_position[0]
    y_diff = position_vector[1] - other_position[1]
    if abs(x_diff) > 500:
        new_x = other_position[0] + 1000 * x_diff / abs(x_diff)
    else:
        new_x = other_position[0]
    if abs(y_diff) > 400:
        new_y = other_position[1] + 800 * y_diff / abs(y_diff)
    else:
        new_y = other_position[1]
    return [new_x, new_y]


def find_relative_heading(
    position_vector: list[float, float],
    other_position: list[float, float],
    ship_heading: float,
) -> float:
    heading = np.arctan2(
        other_position[0] - position_vector[0], other_position[1] - position_vector[1]
    )
    heading = 90 - (heading * 180 / np.pi)
    while heading < 0:
        heading += 360
    rel_head = ship_heading - heading
    while rel_head < 0:
        rel_head += 360
    return rel_head


def find_desired_angle(
    position_vector: list[float, float],
    speed: float,
    other_position: list[float, float],
    other_velocity: list[float, float],
    extrapolate: bool,
) -> float:
    """
    Finds angle ship should be at when trying to shoot a specific asteroid
    Returned angle is 0-360 going counter-clockwise with 0 in the positive y direction
    """
    # This line makes the desired angle be for the next timestep of the asteroid to avoid lagging behind
    if extrapolate:
        velocity_increment = [x / 30 for x in other_velocity]
        other_position = [sum(x) for x in zip(other_position, velocity_increment)]
    a = -other_velocity[1]
    b = position_vector[0] - other_position[0]
    c = speed
    d = -other_velocity[0]
    f = position_vector[1] - other_position[1]
    theta = 2 * np.arctan(
        (
            b * c
            - np.sqrt(
                -(a**2 * b**2)
                + 2 * a * b * d * f
                + b**2 * c**2
                + f**2 * (c**2 - d**2)
            )
        )
        / (a * b - f * (c + d))
    )
    desired_angle = 180 + (theta * 180 / np.pi)
    return desired_angle


if __name__ == "__main__":
    position_vector = [300, 500]
    speed = 50 * 2 ** (0.5)
    other_position = [300, 550]
    other_velocity = [50, 0]
    angle = find_desired_angle(position_vector, speed, other_position, other_velocity)
    print(angle)
