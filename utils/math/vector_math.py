import numpy as np

def calc_intercept_angle(ship_pos, bullet_speed, asteroid_pos, asteroid_velocity):
    # Calculating with a stationary ship assumption

    # Calculate the relative position of the asteroid to the ship
    dx,dy = asteroid_pos[0] - ship_pos[0], asteroid_pos[1] - ship_pos[1]
    # Calculate the asteroid's velocity vector


    # Quadratic formula to solve for the time of intercept
    a = asteroid_velocity[0]**2 + asteroid_velocity[1]**2 - bullet_speed ** 2
    b = 2 * (dx * asteroid_velocity[0] + dy * asteroid_velocity[1])
    c = dx ** 2 + dy ** 2
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return None

    sqrt_disc = np.sqrt(discriminant)

    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    valid_t = [t for t in [t1, t2] if t >= 0]
    t_min = min(valid_t)
    intercept_dx = dx + asteroid_velocity[0] * t_min
    intercept_dy = dy + asteroid_velocity[1] * t_min

    intercept_angle = np.arctan2(intercept_dy, intercept_dx) * 180 / np.pi
    intercept_angle = (intercept_angle) % 360

    return intercept_angle

angle = calc_intercept_angle([0,0], 1.41, [0,1],  [1,0])

def turn_angle(ship_pos, ship_heading, ship_turn_rate, bullet_speed, asteroid_pos, asteroid_velocity, dt):
    print("start of turn angle")
    angle_delta = calc_intercept_angle(ship_pos, bullet_speed, asteroid_pos, asteroid_velocity) - ship_heading
    print("angle delta", angle_delta)
    if 0 <= (angle_delta) < 180:
        right_turn = False
    else:
        right_turn = True

    if right_turn:
        right_turn_rate = ship_turn_rate[1]

        if angle_delta > right_turn_rate * dt:
            turn_rate = right_turn_rate
            print("1")
        else:
            turn_rate = angle_delta
            print("2")

    else:
        left_turn_rate = ship_turn_rate[0]

        if angle_delta < left_turn_rate * dt:
            turn_rate = left_turn_rate
            print("3")
        else:
            turn_rate = angle_delta
            print("4")

    print("end of turn angle")
    return turn_rate