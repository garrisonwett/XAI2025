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
