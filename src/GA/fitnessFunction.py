import numpy as np
import sys
from src.fuzzy_tools import CustomFIS
from fuzzy_asteroids.fuzzy_asteroids import AsteroidGame, FuzzyAsteroidGame
from fuzzy_asteroids.fuzzy_asteroids import TrainerEnvironment


def AsteriodFitness(chrom, game):
    score = game.run(controller=FuzzyController(chrom), score=SampleScore())
    return score.fitness
