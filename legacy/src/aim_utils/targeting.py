import numpy as np
from src.vector_utils.trajectories import *
from src.fuzzy_tools.CustomFIS import HeiTerry_FIS


class FuzzyThreatAssess:
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
        distance = min(1.0, (distance - 20) / 1000)
        closure_rate = closure_rate / 15
        if closure_rate > 1.0:
            closure_rate = 1.0
        elif closure_rate < -1.0:
            closure_rate = -1.0
        ins = [["dist_to", distance], ["closure_rate", closure_rate]]
        threat = self.FIS.compute(ins, "threat")
        return threat
