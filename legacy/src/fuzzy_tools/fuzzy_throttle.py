import numpy as np
from src.vector_utils.trajectories import *
from src.fuzzy_tools.CustomFIS import HeiTerry_FIS


class FuzzyThreatArea:
    def __init__(self, chromosome) -> None:
        self.FIS = HeiTerry_FIS()
        self.FIS.add_input("dist_to", [0, 1], 3)
        self.FIS.add_input("closure_rate", [-1, 1], 3)
        self.FIS.add_output("thrust", [-1, 1], 3)
        rule_base = chromosome
        v_str = ["dist_to", "closure_rate", "thrust"]
        mfs3 = ["0", "1", "2"]
        # Find a way to automate finding num rules per input earlier and for num inputs
        rules_all = []
        for wow in range(3):
            for gee in range(3):
                rules_all.append(
                    [
                        [[v_str[0], mfs3[wow]], [v_str[1], mfs3[gee]]],
                        ["AND"],
                        [[v_str[2], str(rule_base[(wow) + gee])]],
                    ]
                )

        self.FIS.generate_mamdani_rule(rules_all)

    def area_thrust(self, directionality):
        ins = [["directionality", directionality]]
        thrust = self.FIS.compute(ins, "thrust")
        return thrust


# Questions for danny

# 1) What exactly is the directionality variable. Its clearly an array of asteroids in slices relative to the ship
# but is directionality[0] directly off the nose or is it in a fixed direction. If off the nose, what direction does it go, if not
# what is the fixed direction

# 2) What outputs do you want me to use for the fuzzy? Just directionality? Should I include a "panic button" where if an asteroid
# is SUPER CLOSE it just goes away from that and doesnt care about anything else?

# 3) What is v_str, and do i need to change anything about the wow-gee loops?
