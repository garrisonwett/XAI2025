import numpy as np


def distance_to(
    position_vector: list[float, float], other_position: list[float, float]
):
    return (
        (position_vector[0] - other_position[0]) ** 2
        + (position_vector[1] - other_position[1]) ** 2
    ) ** 0.5


def closing_rate(
    position_vector: list[float, float],
    ship_velocity: list[float, float],
    other_position: list[float, float],
    other_velocity: list[float, float],
):
    rel_vel = [x - y for x, y in zip(other_velocity, ship_velocity)]
    asteroid_vec = [x - y for x, y in zip(position_vector, other_position)]
    mag = np.sqrt(asteroid_vec[0]**2 + asteroid_vec[1]**2)
    asteroid_unit_vec = [asteroid_vec[0]/mag, asteroid_vec[1]/mag]
    closure = sum([x * y for x, y in zip(asteroid_unit_vec, rel_vel)])
    return closure


if __name__ == "__main__":
    position_vector = [10, 10]
    other_position = [20, 20]
    other_velocity = [0.5, 0.5]
    print(distance_to(position_vector, other_position))
    print(closing_rate(position_vector, [0,0], other_position, other_velocity))
