import numpy as np
from GA.chromosome import *
import random


def randPointCross(string1, string2):
    leng = len(string1)
    ranInt = random.randint(0, leng - 2)
    ranInt2 = random.randint(ranInt, leng - 1)
    child1 = string1[0:ranInt] + string2[ranInt:ranInt2] + string1[ranInt2:leng]
    child2 = string2[0:ranInt] + string1[ranInt:ranInt2] + string2[ranInt2:leng]
    return child1, child2


def SingleCrossover(parent1, parent2, PC):
    parent1 = parent1.getString()
    p1_rb1 = parent1[0]
    p1_rb2 = parent1[1]

    parent2 = parent2.getString()
    p2_rb1 = parent2[0]
    p2_rb2 = parent2[1]
    child1 = []
    child2 = []

    # crossover rule base
    c1_rb1, c2_rb1 = randPointCross(p1_rb1, p2_rb1)
    c1_rb2, c2_rb2 = randPointCross(p1_rb2, p2_rb2)

    child1 = [c1_rb1, c1_rb2]
    child2 = [c2_rb1, c2_rb2]

    rand = random.random()
    if rand <= PC:
        child1 = Chromosome(child1)
    else:
        child1 = Chromosome(parent1)

    rand = random.random()
    if rand <= PC:
        child2 = Chromosome(child2)
    else:
        child2 = Chromosome(parent2)

    return child1, child2


if __name__ == "__main__":
    p1 = Chromosome([0.205, 0.1106])
    p2 = Chromosome([0.3, 0.6])
    c1, c2 = SingleCrossover(p1, p2, 1)
    print(c1, c2)
