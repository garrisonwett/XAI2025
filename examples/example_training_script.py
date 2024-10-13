# -*- coding: utf-8 -*-

import time
import numpy as np
from kesslergame import Scenario, KesslerGame, GraphicsType
from kesslergame import TrainerEnvironment
from src.FuzzyController import FuzzyController
import time
import threading
import random


scenario_random_repeatable = Scenario(
    name="random_repeatable",
    num_asteroids=12,
    ship_states=[
        {"position": (400, 400), "angle": 90, "lives": 3, "team": 1},
        # {"position": (400, 600), "angle": 90, "lives": 3, "team": 2},
    ],
    map_size=(1000, 800),
    time_limit=120,
    ammo_limit_multiplier=0,
    stop_if_no_ammo=False,
    seed=1,
)

scenario_asteroid_wall = Scenario(
    name="asteroid_wall",
    asteroid_states=[
        {"position": (200, 200), "angle": 90.0, "speed": 40},
        {"position": (300, 200), "angle": 90.0, "speed": 40},
        {"position": (400, 200), "angle": 90.0, "speed": 40},
        {"position": (500, 200), "angle": 90.0, "speed": 40},
        {"position": (600, 200), "angle": 90.0, "speed": 40},
        {"position": (100, 200), "angle": 90.0, "speed": 40},
        {"position": (700, 200), "angle": 90.0, "speed": 40},
        {"position": (800, 200), "angle": 90.0, "speed": 40},
        {"position": (900, 200), "angle": 90.0, "speed": 40},
    ],
    ship_states=[{"position": (400, 500)}],
)

scenario_tracking_test = Scenario(
    name="tracking_test",
    asteroid_states=[
        {"position": (100, 200), "angle": 100.0, "speed": 200},
    ],
    ship_states=[{"position": (500, 600)}],
)

# It fucking Sucks at this one lol
scenario_diversion = Scenario(
    name="diversion",
    asteroid_states=[
        {"position": (200, 200), "angle": 110.0, "speed": 40},
        {"position": (300, 200), "angle": 110.0, "speed": 40},
        {"position": (500, 100), "angle": 90.0, "speed": 40},
        {"position": (700, 200), "angle": 70.0, "speed": 40},
        {"position": (800, 200), "angle": 70.0, "speed": 40},
    ],
    ship_states=[{"position": (500, 500)}],
)

scenario_cross = Scenario(
    name="cross",
    asteroid_states=[
        {"position": (400, 700), "angle": 271.0, "speed": 40},
        {"position": (400, 100), "angle": 91.0, "speed": 40},
        {"position": (800, 400), "angle": 181.0, "speed": 40},
        {"position": (100, 400), "angle": 1.0, "speed": 40},
    ],
    ship_states=[{"position": (400, 400)}],
)

portfolio = [
    scenario_asteroid_wall,
    scenario_cross,
    scenario_tracking_test,
    scenario_diversion,
]

game_settings = {
    "perf_tracker": True,
    "realtime_multiplier": 0,
}
game = TrainerEnvironment(
    settings=game_settings
)  # Use this for max-speed, no-graphics simulation


def fit_func(chrom, seed_in):
    pre = time.perf_counter()
    chrom = [int(x) for x in chrom]
    asteroids_hit_total = 0
    deaths = 0
    eval_time = 0
    for each_scenario in portfolio:
        score, perf_data = game.run(
            scenario=each_scenario,
            controllers=[FuzzyController(chrom), FuzzyController(chrom)],
        )

        print("Scenario eval time: " + str(time.perf_counter() - pre))
        print(score.stop_reason)
        print(chrom)
        print("Asteroids hit: " + str([team.asteroids_hit for team in score.teams]))
        print("Deaths: " + str([team.deaths for team in score.teams]))
        print("Accuracy: " + str([team.accuracy for team in score.teams]))
        print("Mean eval time: " + str([team.mean_eval_time for team in score.teams]))
        asteroids_hit_total += [team.asteroids_hit for team in score.teams][0]
        accuracy = [team.accuracy for team in score.teams][0]
        deaths += [team.deaths for team in score.teams][0] * 100
        eval_time += [team.mean_eval_time for team in score.teams][0] * 100
    return asteroids_hit_total - deaths  # - eval_time


def initialize(PopSize, IndivSize):
    pop = np.random.randint(3, size=(PopSize, IndivSize))
    return pop


def selection(pop, fit_arr):
    M = len(pop)
    fit_arr = np.asarray(fit_arr)
    normalized_fitness = fit_arr / sum(fit_arr)

    norm_idx = np.argsort(normalized_fitness)
    norm_idx = np.flip(norm_idx)

    normalized_fitness = np.sort(normalized_fitness)
    normalized_fitness = np.flip(normalized_fitness, 0)

    tempPop = pop

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
    return parent1, parent2


def crossover(parent1, parent2, PC):
    rand = random.random()
    if rand <= PC:
        leng = len(parent1)
        ranInt = random.randint(1, leng - 1)
        child1 = np.append(parent1[0:ranInt], parent2[ranInt:leng])
        child2 = np.append(parent1[0:ranInt], parent1[ranInt:leng])
        return child1, child2
    else:
        return parent1, parent2


def mutation(child, PM):
    rand = random.random()
    for i in range(len(child)):
        if rand <= PM:
            child[i] = random.randint(0, 2)
    return child


def elitism(pop, newPop, fit_arr, ER):
    M = len(newPop)
    N = len(newPop[0])
    Elite_no = round(M * ER)
    all = np.vstack((pop, newPop))
    idx = np.argsort(fit_arr)
    idx = np.flip(idx)
    newPop2 = np.zeros((M, N))
    fit_array = np.zeros(M)
    for k in range(Elite_no):
        newPop2[k] = all[idx[k]]
        fit_array[k] = fit_arr[idx[k]]
    for k in np.arange(Elite_no, M, 1):
        newPop2[k] = all[random.randint(0, 2 * M - 1)]
        fit_array[k] = fit_arr[random.randint(0, 2 * M - 1)]
    return newPop2, fit_array


class GA:
    def __init__(self, IndivSize, PopSize, maxGen, PC, PM, ER):
        self.IndivSize = IndivSize
        self.PopSize = PopSize
        self.maxGen = maxGen
        self.PC = PC
        self.PM = PM
        self.ER = ER

    def run(self):
        self.population = initialize(self.PopSize, self.IndivSize)
        fit_plot = []
        for i in range(self.maxGen):
            fit_arr = [fit_func(gains, 0) for gains in self.population]
            print("Generation: " + str(i))
            # print("Best Fitness: " + str(self.getBest().getFitness()))
            newPop = []
            a = np.arange(0, self.PopSize, 2)
            a = a.tolist()
            for k in a:
                parent1, parent2 = selection(self.population, fit_arr)
                child1, child2 = crossover(parent1, parent2, self.PC)
                child1 = mutation(child1, self.PM)
                child2 = mutation(child2, self.PM)
                newPop.append(child1)
                newPop.append(child2)
            fit_plot.append(self.getBest(self.population, fit_arr)[1])
            new_fit = np.asarray([fit_func(gains, i) for gains in newPop])
            fit_arr_big = np.append(fit_arr, new_fit)
            newPop, fit_arr = elitism(self.population, newPop, fit_arr_big, self.ER)
            self.population = newPop
        return self.getBest(self.population, fit_arr), fit_plot

    def getBest(self, pop, fit_arr):
        norm_idx = np.argmax(fit_arr)
        return pop[norm_idx], fit_arr[norm_idx]

#232, 232, 232, 232, 232, 232, 232, 232, 232, 364, 364, 364, 364, 364, 364, 364, 364, 364, 478, 478, 478, 478, 478, 478, 478, 478, 480, 480, 480, 480
# IndivSize, PopSize, maxGen, PC, PM, ER
ga = GA(18, 10, 10, 0.9, 0.3, 0.1)
best, fit_plot = ga.run()
print(best)
print(fit_plot)
