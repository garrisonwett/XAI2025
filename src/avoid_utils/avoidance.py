import numpy as np
from src.vector_utils.trajectories import *
from src.fuzzy_tools.CustomFIS import HeiTerry_FIS
import math
import sys


class AvoidThreatAssess:
    def __init__(self, chromosome) -> None:
        self.FIS = HeiTerry_FIS()
        self.FIS.add_input("dist_to", [0, 1], 3)
        self.FIS.add_input("closure_rate", [-1, 1], 3)
        self.FIS.add_output("threat", [0, 1], 3)
        rule_base = chromosome
        v_str = ["dist_to", "closure_rate", "threat"]
        mfs3 = ["0", "1", "2", "3", "4", "5", "6"]
        # Find a way to automate finding num rules per input earlier and for num inputs
        rules_all = []
        i = 0
        for wow in range(3):
            for gee in range(3):
                rules_all.append(
                    [
                        [[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]]],
                        ["AND"],
                        [[v_str[2], str(rule_base[i])]],
                    ]
                )
                i += 1
        self.FIS.generate_mamdani_rule(rules_all)

    def assign_fuzzy_threat(self, distance, closure_rate):
        distance = min(1.0, (distance - 50) / 1000)
        if distance < 0.0:
            distance = 0.0
        closure_rate = closure_rate / 10000
        if closure_rate > 1.0:
            closure_rate = 1.0
        elif closure_rate < -1.0:
            closure_rate = -1.0
        ins = [["dist_to", distance], ["closure_rate", closure_rate]]
        threat = self.FIS.compute(ins, "threat")
        return threat


def future_position(position, heading, speed, turn_rate, thrust):
    delta_time = 1 / 30
    drag = 80
    max_speed = 240
    drag_amount = drag * delta_time
    if drag_amount > abs(speed):
        speed = 0
    else:
        speed -= drag_amount * np.sign(speed)
    speed += thrust * delta_time
    if speed > max_speed:
        speed = max_speed
    elif speed < -max_speed:
        speed = -max_speed
    heading += turn_rate * delta_time
    while heading > 360:
        heading -= 360.0
    while heading < 0:
        heading += 360.0
    velocity = [
        math.cos(math.radians(heading)) * speed,
        math.sin(math.radians(heading)) * speed,
    ]
    position = [pos + v * delta_time for pos, v in zip(position, velocity)]
    return position

def mid_gap_angles(a):
    n = len(a)
    max1 = -sys.maxsize - 1
    for i in range(0, n, 1):
        for j in range(0, n, 1):
            new_max = abs(a[i] - a[j])
            if new_max > 180: new_max += -180
            if new_max > max1:
                max1 = new_max
                fin = (a[i] + a[j])/2
                # print(f"{a[i]} and {a[j]}")
    return fin