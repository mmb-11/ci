from deap import base, creator, tools, algorithms
import random
import numpy as np

# Evaluation function: minimize x^2
def evaluate(ind):
    return ind[0] ** 2,

# Set up the problem: minimize fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=50)

# Run GA
population, logbook  = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, verbose=True)

best = tools.selBest(population, 1)[0]
print(f"Best x: {best[0]:.4f}, f(x): {evaluate(best)[0]:.4f}")
