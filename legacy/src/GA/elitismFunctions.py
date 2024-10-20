import numpy as np
from GA.chromosome import *
import random


def basicElitism(population, newPop, Er):
    M = len(newPop)
    Elite_no = round(M * Er)

    all = population + newPop
    fitnessArr = np.array(list(map(lambda a: a.getFitness(), all)))

    idx = np.argsort(fitnessArr)
    idx = np.flip(idx)
    newPop2 = np.zeros(M)
    newPop2 = newPop2.tolist()
    for k in range(Elite_no):
        newPop2[k] = all[idx[k]]

    for k in np.arange(Elite_no, M, 1):
        newPop2[k] = all[k - Elite_no]

    return newPop2


def ElitismTest(population, newPop, Er):
    newPop = basicElitism(population, newPop, Er)
    fitnessArr = np.array(list(map(lambda a: a.getFitness(), population)))
    norm_idx = np.argsort(fitnessArr)
    norm_idx = np.flip(norm_idx)
    bestChrom = population[norm_idx[0]]
    newPop.pop()
    newPop.append(bestChrom)

    return newPop


if __name__ == "__main__":
    chrom = Chromosome(np.array([3, 9, 7, 8, 9, 10]))
    chrom2 = Chromosome(np.array([1, 2, 3, 4, 5, 6]))
    chrom3 = Chromosome(np.array([1, 2, 3, 4, 5, 6]))
    chrom4 = Chromosome(np.array([1, 2, 3, 4, 5, 6]))
    chrom.updateFitness(8)
    chrom2.updateFitness(4)
    chrom3.updateFitness(5)
    chrom4.updateFitness(-5)
    population = [chrom, chrom2, chrom3, chrom4]
    chrom = Chromosome(np.array([5, 6, 7, 8, 9, 10]))
    chrom2 = Chromosome(np.array([1, 2, 3, 4, 5, 6]))
    chrom3 = Chromosome(np.array([1, 2, 3, 4, 5, 6]))
    chrom4 = Chromosome(np.array([1, 2, 3, 4, 5, 6]))
    chrom.updateFitness(10)
    chrom2.updateFitness(12)
    chrom3.updateFitness(9)
    chrom4.updateFitness(-9)
    newpopulation = [chrom, chrom2, chrom3, chrom4]
    e = ElitismTest(population, newpopulation, 0.2)
    for chrom in e:
        print(chrom, "\n")
