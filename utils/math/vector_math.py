import numpy as np

def calc_intercept_angle(ship_pos, bullet_speed, asteroid_pos, asteroid_speed, asteroid_heading):
    # Calculating with a stationary ship assumption

    # Calculate the relative position of the asteroid to the ship
    dx,dy = asteroid_pos[0] - ship_pos[0], asteroid_pos[1] - ship_pos[1]
    print(dx,dy)
    # Calculate the asteroid's velocity vector
    asteroid_heading_rad = np.pi / 180 * asteroid_heading
    asteroid_velocity_vector = [asteroid_speed * np.cos(asteroid_heading_rad), asteroid_speed * np.sin(asteroid_heading_rad)]

    # Quadratic formula to solve for the time of intercept
    a = asteroid_velocity_vector[0]**2 + asteroid_velocity_vector[1]**2 - bullet_speed ** 2
    b = 2 * (dx * asteroid_velocity_vector[0] + dy * asteroid_velocity_vector[1])
    c = dx ** 2 + dy ** 2
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return None

    sqrt_disc = np.sqrt(discriminant)

    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    valid_t = [t for t in [t1, t2] if t >= 0]
    t_min = min(valid_t)
    intercept_dx = dx + asteroid_velocity_vector[0] * t_min
    intercept_dy = dy + asteroid_velocity_vector[1] * t_min

    intercept_angle = np.arctan2(intercept_dy, intercept_dx) * 180 / np.pi
    intercept_angle = (intercept_angle) % 360
    return intercept_angle

# angle = calc_intercept_angle([0,0], 2, [a,b],  0, 0)
# print(angle)
