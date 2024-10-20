import numpy as np
from src.vector_utils.trajectories import *


def angle_to_speed(angle_desired, current_angle) -> float:
    # turn rate of 30 equals 1 degree per time step ccw... this may be the framerate
    rate = 30 * (angle_desired - current_angle)
    return rate


def speed_to_thrust(speed_desired, current_speed) -> float:
    thr = 30 * (speed_desired - current_speed)
    return thr