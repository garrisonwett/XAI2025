import numpy as np
import random
from GA.chromosome import *


def asteriodInitialize(numbChroms):
    population = []
    for i in range(numbChroms):
        population.append(Chromosome(np.random.randint(-3, high=10, size=(3)).tolist()))
    return population


if __name__ == "__main__":
    asteriodInitialize(10)
