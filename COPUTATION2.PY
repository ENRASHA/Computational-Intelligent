
import numpy as np

# Holder Table Function
def holder_table_function(x):
    return -abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(abs(1 - np.sqrt(x[0]**2 + x[1]**2) / np.pi)))

# Fitness Function
def fitness(individual):
    return holder_table_function(individual)

# Initialize Population
def initialize_population(pop_size, dimensions, bounds):
    return np.random.uniform(bounds[0], bounds[1], (pop_size, dimensions))

# Selection (Tournament Selection)
def selection(population, fitnesses, tournament_size):
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    selected_fitnesses = fitnesses[selected_indices]
    winner_index = selected_indices[np.argmin(selected_fitnesses)]
    return population[winner_index]