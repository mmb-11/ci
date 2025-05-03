!pip install deap scikit-learn numpy

import numpy as np
from sklearn.neural_network import MLPRegressor
from deap import base, creator, tools, algorithms

# Train sample neural network
X = np.array([[150, 30, 3], [160, 35, 4], [170, 40, 2]])
y = np.array([0.85, 0.90, 0.80])
model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000).fit(X, y)

# Fitness function
def evaluate(ind):
    return model.predict([ind])[0],

# Genetic algorithm setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Parameters: inlet temp (100–200), flow rate (10–50), pressure (1–5)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (lambda: np.random.uniform(100, 200),
                  lambda: np.random.uniform(10, 50),
                  lambda: np.random.uniform(1, 5)), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run GA
pop = toolbox.population(n=30)
algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=20, verbose=True)

# Output best result
best = tools.selBest(pop, k=1)[0]
print("Best Parameters:", best)
print("Predicted Yield:", round(evaluate(best)[0], 4))
