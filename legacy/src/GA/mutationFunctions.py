import numpy as np
from GA.chromosome import *
import random


def asteroidMutation(chromosome, PM):
    child = chromosome.getString()
    leng = len(child)
    for x in range(leng):
        for i in range(x):
            R = random.random()
            if R < PM:
                child[x][i] = random.randint(0, 2)
    return Chromosome(child)


if __name__ == "__main__":
    p1 = Chromosome([0.205, 0.1106])
    c1 = asteroidMutation(p1, 1)
    print(c1)
