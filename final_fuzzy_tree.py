import numpy as np

def triangle(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)

def fuzzy_eval(x1, x2, m1, m2, constants):
    low1 = triangle(x1, -0.2, 0.0, m1)
    med1 = triangle(x1, 0.0, m1, 1.0)
    high1 = triangle(x1, m1, 1.0, 1.2)
    low2 = triangle(x2, -0.2, 0.0, m2)
    med2 = triangle(x2, 0.0, m2, 1.0)
    high2 = triangle(x2, m2, 1.0, 1.2)
    levels = [low1, med1, high1]
    levels2 = [low2, med2, high2]
    num = 0.0
    den = 0.0
    idx = 0
    for a in range(3):
        for b in range(3):
            w = levels[a] * levels2[b]
            num += w * constants[idx]
            den += w
            idx += 1
    if den == 0:
        return 0.0
    return num / den

def evaluate_fuzzy_tree(inputs):
    x1 = (
        return inputs[0]
    )
    x2 = (
        return inputs[1]
    )
    medium1 = 0.693920054202662
    medium2 = 0.10391289526858849
    constants = [0.3166918153879191, 0.794022060202132, -0.1619503664985903, -0.20068585182910703, 0.3491336890563389, 0.28291958304958875, 1.092863689873111, 0.8628813505750359, 1.049514591962747]
    return fuzzy_eval(x1, x2, medium1, medium2, constants)
