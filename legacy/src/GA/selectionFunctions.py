import numpy as np
from GA.chromosome import *
import random


def basicSelection(population):
    M = len(population)
    fitnessArr = np.array(list(map(lambda a: a.getFitness(), population)))
    if any(list(map(lambda a: a.getFitness() < 0, population))):
        b = np.abs(min(fitnessArr))
        Scaled_fitness = fitnessArr + b
        normalized_fitness = Scaled_fitness / sum(Scaled_fitness)
    else:
        normalized_fitness = fitnessArr / sum(fitnessArr)

    norm_idx = np.argsort(normalized_fitness)
    norm_idx = np.flip(norm_idx)

    normalized_fitness = np.sort(normalized_fitness)
    normalized_fitness = np.flip(normalized_fitness, 0)

    tempPop = []
    for i in range(M):
        Chrom = Chromosome(population[norm_idx[i]].getString())
        Chrom.updateFitness(population[norm_idx[i]].getFitness())
        Chrom.updateNormFitness(normalized_fitness[norm_idx[i]])
        tempPop.append(Chrom)

    cumsum = np.zeros(M)
    for i in range(M):
        cumsum[i] = sum(normalized_fitness[i:])

    R = random.random()
    parent1_idx = M - 1
    for i in range(len(cumsum)):
        if R > cumsum[i]:
            parent1_idx = i - 1
            break

    parent2_idx = parent1_idx
    break_loop = 0
    while parent2_idx == parent1_idx:
        break_loop += 1
        R = random.random()
        if break_loop > 20:
            break

        for i in range(len(cumsum)):
            if R > cumsum[i]:
                parent2_idx = i - 1
                break

    parent1 = tempPop[parent1_idx]
    parent2 = tempPop[parent2_idx]
    # print(parent1_idx,parent2_idx)
    return parent1, parent2


if __name__ == "__main__":
    chrom = Chromosome(np.array([5, 6, 7, 8, 9, 10]))
    chrom2 = Chromosome(np.array([1, 2, 3, 4, 5, 6]))
    chrom3 = Chromosome(np.array([1, 2, 3, 4, 5, 6]))
    chrom4 = Chromosome(np.array([1, 2, 3, 4, 5, 6]))
    chrom.updateFitness(8)
    chrom2.updateFitness(4)
    chrom3.updateFitness(5)
    chrom4.updateFitness(-5)
    population = [chrom, chrom2, chrom3, chrom4]
    basicSelection(population)
