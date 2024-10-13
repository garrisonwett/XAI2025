import numpy as np
import matplotlib as mpl
import random
from GA.fitnessFunction import *
from GA.chromosome import *
from GA.initializationFunctions import *
from GA.selectionFunctions import *
from GA.crossoverFunctions import *
from GA.mutationFunctions import *
from GA.elitismFunctions import *
import threading


class CGA:
    def __init__(self, NumberOfChrom, NumbofGenes, maxGen, PC, PM, Er, bounds):
        self.numbChroms = NumberOfChrom
        self.numbGenes = NumbofGenes
        self.maxGen = maxGen
        self.PC = PC
        self.PM = PM
        self.population = []
        self.Er = Er
        self.bounds = bounds

    def run(
        self,
        initializationFunction,
        selectionFunction,
        crossoverFunction,
        fitnessFunction,
        mutationFunction,
        elitismFunction,
    ):
        self.population = initializationFunction(
            self.numbChroms, self.numbGenes, self.bounds
        )
        for chrom in self.population:
            print(chrom, "\n")
        for i in range(self.maxGen):
            print("Generation: " + str(i))
            print("Best Fitness: " + str(self.getBestChromosome().getFitness()))
            # obtain fitness values
            for chrom in self.population:
                chrom.updateFitness(fitnessFunction(chrom, self.bounds))
            newPop = []
            a = np.arange(0, self.numbChroms, 2)
            a = a.tolist()
            for k in a:
                # crossover
                parent1, parent2 = selectionFunction(self.population)
                child1, child2 = crossoverFunction(
                    parent1, parent2, self.numbGenes, self.PC, self.bounds
                )

                # mutation
                child1 = mutationFunction(child1, self.numbGenes, self.PM, self.bounds)
                child2 = mutationFunction(child2, self.numbGenes, self.PM, self.bounds)

                newPop.append(child1)
                newPop.append(child2)

            # update fitness values
            for chrom in newPop:
                chrom.updateFitness(fitnessFunction(chrom, self.bounds))

            # elitism
            newPop = elitismFunction(self.population, newPop, self.Er)

            self.population = newPop

    def getBestChromosome(self):
        fitnessArr = np.array(list(map(lambda a: a.getFitness(), self.population)))
        norm_idx = np.argsort(fitnessArr)
        norm_idx = np.flip(norm_idx)
        bestChrom = self.population[norm_idx[0]]
        return bestChrom


class CGAThread:
    def __init__(self, NumberOfChrom, NumbofGenes, maxGen, PC, PM, Er, bounds):
        self.numbChroms = NumberOfChrom
        self.numbGenes = NumbofGenes
        self.maxGen = maxGen
        self.PC = PC
        self.PM = PM
        self.population = []
        self.Er = Er
        self.bounds = bounds

    def run(
        self,
        initializationFunction,
        selectionFunction,
        crossoverFunction,
        fitnessFunction,
        mutationFunction,
        elitismFunction,
    ):
        self.population = initializationFunction(
            self.numbChroms, self.numbGenes, self.bounds
        )
        for chrom in self.population:
            print(chrom, "\n")
        for i in range(self.maxGen):
            print("Generation: " + str(i))
            print("Best Fitness: " + str(self.getBestChromosome().getFitness()))
            for chrom in self.population:
                chrom.updateFitness(fitnessFunction(chrom, self.bounds))
            newPop = []
            a = np.arange(0, self.numbChroms, 2)
            a = a.tolist()

            for k in a:
                # crossover
                parent1, parent2 = selectionFunction(self.population)
                child1, child2 = crossoverFunction(
                    parent1, parent2, self.numbGenes, self.PC, self.bounds
                )

                # mutation
                child1 = mutationFunction(child1, self.numbGenes, self.PM, self.bounds)
                child2 = mutationFunction(child2, self.numbGenes, self.PM, self.bounds)

                newPop.append(child1)
                newPop.append(child2)

            def ChromThread(chrom):
                chrom.updateFitness(fitnessFunction(chrom, self.bounds))

            # update fitness values
            threads = []
            for chrom in newPop:
                t = threading.Thread(target=ChromThread, args=(chrom,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()
                # chrom.updateFitness(fitnessFunction(chrom, self.bounds))

            # elitism
            newPop = elitismFunction(self.population, newPop, self.Er)

            self.population = newPop

    def getBestChromosome(self):
        fitnessArr = np.array(list(map(lambda a: a.getFitness(), self.population)))
        norm_idx = np.argsort(fitnessArr)
        norm_idx = np.flip(norm_idx)
        bestChrom = self.population[norm_idx[0]]
        return bestChrom


# Test
if __name__ == "__main__":
    # CGA = CGA(NumberOfChrom = 30,
    #           NumbofGenes = 2,
    #           maxGen = 2000,
    #           PC = 0.85,
    #           PM = 0.15,
    #           Er = 0.2)

    # CGA.initialization(CGAInitialize)
    # CGA.run(selectionFunction = basicSelection,
    #         crossoverFunction = basicCrossover,
    #         mutationFunction = basicMutation,
    #         fitnessFunction = fuzzyHomeworkFitness,
    #         elitismFunction = basicElitism)
    # best = CGA.getBestChromosome()
    pass
