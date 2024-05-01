import pandas as pd 
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

# Crossover (Uniform Crossover)
def crossover(parent1, parent2, crossover_rate):
    child = parent1.copy()
    for i in range(len(child)):
        if np.random.rand() < crossover_rate:
            child[i] = parent2[i]
    return child

# Mutation
def mutation(individual, mutation_rate, bounds):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.uniform(bounds[0], bounds[1])
    return individual


# Genetic Algorithm
def genetic_algorithm(pop_size, dimensions, bounds, generations, tournament_size, crossover_rate, mutation_rate):
    population = initialize_population(pop_size, dimensions, bounds)
    best_fitness = float('inf')
    best_individual = None

    for generation in range(generations):
        fitnesses = np.array([fitness(ind) for ind in population])
        new_population = []
        for _ in range(pop_size):
            parent1 = selection(population, fitnesses, tournament_size)
            parent2 = selection(population, fitnesses, tournament_size)
            child = crossover(parent1, parent2, crossover_rate)
            child = mutation(child, mutation_rate, bounds)
            new_population.append(child)

        population = np.array(new_population)
        best_index = np.argmin(fitnesses)
        if fitnesses[best_index] < best_fitness:
            best_fitness = fitnesses[best_index]
            best_individual = population[best_index]

    return best_individual, best_fitness

# Parameters
pop_size = 100
dimensions = 2
bounds = (-10, 10)
generations = 100
tournament_size = 5
crossover_rate = 0.7
mutation_rate = 0.1
# Run Genetic Algorithm
best_solution, best_fitness = genetic_algorithm(pop_size, dimensions, bounds, generations, tournament_size, crossover_rate, mutation_rate)
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)



