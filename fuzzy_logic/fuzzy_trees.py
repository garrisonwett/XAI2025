import math
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def thrust_tree(closure = 0, heading = 0):
    if heading % 1 == 0:
        heading = heading + 0.0001

    # 1. Define the fuzzy variables (Antecedents/Consequents) and their ranges
    closure_rate = ctrl.Antecedent(np.arange(-1000, 1000, 1), 'closure_rate')
    relative_heading = ctrl.Antecedent(np.arange(-45, 315, 1), 'relative_heading')
    output_value = ctrl.Consequent(np.arange(-500, 500, 1), 'output_value')

    closure_rate['negative'] = fuzz.trimf(closure_rate.universe, [-1000, -1000, 0])
    closure_rate['zero'] = fuzz.trimf(closure_rate.universe, [-1000, 0, 1000])
    closure_rate['positive'] = fuzz.trimf(closure_rate.universe, [0, 1000, 1000])

    relative_heading['forward'] = fuzz.trimf(relative_heading.universe, [-45, 0, 45])
    relative_heading['left'] = fuzz.trimf(relative_heading.universe, [45, 90, 135])
    relative_heading['back'] = fuzz.trimf(relative_heading.universe, [135, 180, 225])
    relative_heading['right'] = fuzz.trimf(relative_heading.universe, [225, 270, 315])

    output_value['negative'] = fuzz.trimf(output_value.universe, [-500, -500, 0])
    output_value['small_negative'] = fuzz.trimf(output_value.universe, [-500, -50, 0])
    output_value['zero'] = fuzz.trimf(output_value.universe, [-500, 0, 500])
    output_value['small_positive'] = fuzz.trimf(output_value.universe, [0, 50, 500])
    output_value['positive'] = fuzz.trimf(output_value.universe, [0, 500, 500])

    # 2. Define the fuzzy rules

    rule1 = ctrl.Rule(closure_rate['negative'] & relative_heading['forward'], output_value['small_negative'])
    rule2 = ctrl.Rule(closure_rate['negative'] & relative_heading['left'], output_value['zero'])
    rule3 = ctrl.Rule(closure_rate['negative'] & relative_heading['back'], output_value['small_positive'])
    rule4 = ctrl.Rule(closure_rate['negative'] & relative_heading['right'], output_value['zero'])

    rule5 = ctrl.Rule(closure_rate['zero'] & relative_heading['forward'], output_value['small_negative'])
    rule6 = ctrl.Rule(closure_rate['zero'] & relative_heading['left'], output_value['zero'])
    rule7 = ctrl.Rule(closure_rate['zero'] & relative_heading['back'], output_value['small_positive'])
    rule8 = ctrl.Rule(closure_rate['zero'] & relative_heading['right'], output_value['zero'])

    rule9 = ctrl.Rule(closure_rate['positive'] & relative_heading['forward'], output_value['negative'])
    rule10 = ctrl.Rule(closure_rate['positive'] & relative_heading['left'], output_value['zero'])
    rule11 = ctrl.Rule(closure_rate['positive'] & relative_heading['back'], output_value['positive'])
    rule12 = ctrl.Rule(closure_rate['positive'] & relative_heading['right'], output_value['zero'])

    # 3. Define the fuzzy system

    thrust_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12])
    thrust = ctrl.ControlSystemSimulation(thrust_ctrl)

    # 4. Pass inputs to the fuzzy system and compute the output
    print(closure, heading)
    thrust.input['closure_rate'] = closure
    thrust.input['relative_heading'] = heading
    thrust.compute()
    return thrust.output['output_value']


# def aim_tree(nearest_10):

import time
import numpy as np


class RuspiniZeroth3():
    def __init__(self, mfs, rules):
        """
        mfs = [[0, c1, c2, ... 1],[0, c1, c2, ... 1],[ 0, c1, c2, ... 1 ],[...]] (center of each triangle)
        rules = [v1, v2, ...]: values for each combo of rules (5x5 mf gives 25 rules)
        """
        mfs[mfs>=1.0] = 0.99999999
        mfs[mfs<=0.0] = 0.00000001
        rules[rules>=1.0] = 0.99999999
        rules[rules<=0.0] = 0.00000001
        zeros = np.zeros((len(mfs[:,0]),1))
        mfs = np.transpose(np.concatenate((zeros, np.cumsum(mfs, axis=1)), axis=1))
        norm_mfs = mfs * (1/np.max(mfs, axis=0))
        self.mfs = np.transpose(norm_mfs)
        self.rules = np.reshape(rules, (np.shape(self.mfs)[1], np.shape(self.mfs)[1], np.shape(self.mfs)[1]))

    def CalcMem(self, ins):
        # Replace values outside with 0 and 1 basically. We also hate 0 and 1 specifically.
        ins = [max(min(0.999999999, i), 0.000000001) for i in ins]
        mems = np.zeros(np.shape(self.mfs))
        for i, mf in enumerate(self.mfs):
            r = mf[0]
            j = 1
            while r <= ins[i]:
                l = mf[j-1]
                r = mf[j]
                j += 1
            mems[i][j-2] = (r - ins[i])/(r-l)
            mems[i][j-1] = (ins[i] - l)/(r-l)
        mem = np.outer(np.outer(mems[0],mems[1]), mems[2]).reshape(len(self.mfs[0]),len(self.mfs[0]),len(self.mfs[0]))
        mr = np.multiply(mem, self.rules)
        return np.sum(mr)/np.sum(mem)


class RuspiniZeroth2():
    def __init__(self, mfs, rules):
        """
        mfs = [[0, c1, c2, ... 1],[0, c1, c2, ... 1],[ 0, c1, c2, ... 1 ],[...]] (center of each triangle)
        rules = [v1, v2, ...]: values for each combo of rules (5x5 mf gives 25 rules)
        """
        # Order centers of functions and change rule order to match indicies
        mfs[mfs>=1.0] = 0.99999999
        mfs[mfs<=0.0] = 0.00000001
        rules[rules>=1.0] = 0.99999999
        rules[rules<=0.0] = 0.00000001
        zeros = np.zeros((len(mfs[:,0]),1))
        mfs = np.transpose(np.concatenate((zeros, np.cumsum(mfs, axis=1)), axis=1))
        norm_mfs = mfs * (1/np.max(mfs, axis=0))
        self.mfs = np.transpose(norm_mfs)
        self.rules = np.reshape(rules, (np.shape(self.mfs)[1], np.shape(self.mfs)[1]))

    def CalcMem(self, ins):
        mems = np.zeros(np.shape(self.mfs))

        if ins[0] <= 0.0: ins[0] = 0.000000000001
        elif ins[0] >= 1.0: ins[0] = 0.999999999999
        if ins[1] <= 0.0: ins[1] = 0.000000000001
        elif ins[1] >= 1.0: ins[1] = 0.999999999999

        for i, mf in enumerate(self.mfs):
            r = mf[0]
            j = 1
            while r <= ins[i]:
                l = mf[j-1]
                r = mf[j]
                j += 1
            mems[i][j-2] = (r - ins[i])/(r-l)
            mems[i][j-1] = (ins[i] - l)/(r-l)
        mem = np.outer(mems[1],mems[0]).reshape(len(self.mfs[0]),len(self.mfs[0]))
        mr = np.multiply(mem, self.rules)
        return np.sum(mr)/np.sum(mem)


if __name__ == "__main__":
    mf_r_flat = [0,0.6,0.4,0.8,1,0,0.4,0.5,0.7,1,0.0,0.1,0.2,0.3,1.0,.5,.6,.7,.8,.9,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.0,0.1,0.2,0.3,0.4,0.0,0.1,0.2,0.3,0.4,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    mfs = np.asarray([mf_r_flat[1:4], mf_r_flat[6:9]])
    rules = np.asarray(mf_r_flat[10:26])

    TestFIS = RuspiniZeroth2(mfs, rules)

    t = time.perf_counter()
    for i in range(1):
        ins = [0.6,0.3]
        output = TestFIS.CalcMem(ins)
    print((time.perf_counter() - t)/1)

    mfs = np.asarray([mf_r_flat[1:4], mf_r_flat[6:9], mf_r_flat[11:14]])
    rules = np.asarray(mf_r_flat[15:79])

    TestFIS = RuspiniZeroth3(mfs, rules)
    t = time.perf_counter()
    for i in range(1):
        ins = [0.6,0.3,0.35]
        output = TestFIS.CalcMem(ins)
    print((time.perf_counter() - t)/1)
