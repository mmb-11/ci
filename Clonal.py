import numpy as np

def clonal_selection(target_fn, pop_size=50, clone_top=10, mutation_rate=0.1, generations=100, dims=2):
    # Create initial random population
    population = np.random.rand(pop_size, dims)

    for gen in range(generations):
        # Calculate fitness for each solution
        fitness = np.array([target_fn(ind) for ind in population])
        
        # Select best solutions (lowest fitness)
        best_indices = np.argsort(fitness)[:clone_top]
        best_clones = population[best_indices]
        
        # Mutate the best clones
        mutated = best_clones + np.random.normal(0, mutation_rate, best_clones.shape)
        mutated = np.clip(mutated, 0, 1)  # Keep within [0, 1]
        
        # Replace worst individuals with mutated ones
        population[:clone_top] = mutated

        # Print best of current generation
        gen_best_idx = np.argmin(fitness)
        gen_best_solution = population[gen_best_idx]
        gen_best_fitness = fitness[gen_best_idx]
        print(f"Generation {gen+1}: Fitness = {gen_best_fitness:.6f}, Solution = {gen_best_solution}")

    # Final best solution
    final_fitness = np.array([target_fn(ind) for ind in population])
    best_idx = np.argmin(final_fitness)
    return population[best_idx], final_fitness[best_idx]

# Example function to minimize (distance to origin)
def target_function(x):
    return np.sum(x**2)

# Run the algorithm
best_solution, best_fitness = clonal_selection(target_function)

print("\nFinal Result:")
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
