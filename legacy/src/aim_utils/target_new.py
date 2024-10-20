import numpy as np
from src.vector_utils.trajectories import *
from src.fuzzy_tools.CustomFIS import HeiTerry_FIS


from fuzzylogic.classes import Domain, Set, Rule
from fuzzylogic.hedges import very
from fuzzylogic.functions import triangular


class FuzzyThreatAssess:
    def __init__(self, chromosome) -> None:
        self.dist_to = Domain("Distance", 0, 1)
        self.closure_rate = Domain("ClosureRate", -1, 1)
        self.threat = Domain("Threat", 0, 1)

        self.dist_to.zero = triangular(-0.5, 0.5, c=0.0)
        self.dist_to.one = triangular(0, 1, c=0.5)
        self.dist_to.two = triangular(0.5, 1.5, c=1.0)

        self.closure_rate.zero = triangular(-2.0, 0.5, c=-1.0)
        self.closure_rate.one = triangular(-1.0, 1.0, c=0.0)
        self.closure_rate.two = triangular(0.0, 2.0, c=1.0)

        self.threat.zero = triangular(-0.5, 0.5, c=0.0)
        self.threat.one = triangular(0.0, 1.0, c=0.5)
        self.threat.two = triangular(0.5, 1.5, c=1.0)

        v_str = ["self.dist_to", "self.closure_rate", "self.threat"]
        mfs = ["zero", "one", "two", "three", "four", "five", "six"]
        # Find a way to automate finding num rules per input earlier and for num inputs
        rule_base = chromosome
        rules_all = {}
        for wow in range(3):
            for gee in range(3):
                rules_all[
                    eval(v_str[0] + "." + mfs[wow]), eval(v_str[1] + "." + mfs[gee])
                ] = eval(v_str[2] + "." + mfs[rule_base[wow + gee]])
        self.rules = Rule(rules_all)

    def assign_fuzzy_threat(self, distance, closure_rate):
        distance = min(1.0, distance / 1000 - 20 / 1000)
        closure_rate = closure_rate / 10000
        if closure_rate > 1.0:
            closure_rate = 1.0
        elif closure_rate < -1.0:
            closure_rate = -1.0
        values = {self.dist_to: distance, self.closure_rate: closure_rate}
        return self.rules(values)
