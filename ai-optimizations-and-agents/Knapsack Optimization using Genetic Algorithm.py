import numpy as np
import pandas as pd
import random as rd
from random import randint
import matplotlib.pyplot as plt

# Initialize problem data
item_number = np.arange(1, 11)
weight = np.random.randint(1, 15, size=10)
value = np.random.randint(10, 750, size=10)
knapsack_threshold = 35  # Maximum weight the knapsack can hold

# Display items
print('The list is as follows:')
print('Item No.   Weight   Value')
for i in range(item_number.shape[0]):
    print(f'{item_number[i]:<10} {weight[i]:<10} {value[i]}')

# Genetic algorithm parameters
solutions_per_pop = 8
pop_size = (solutions_per_pop, item_number.shape[0])
num_generations = 50
num_parents = int(solutions_per_pop / 2)
num_offsprings = solutions_per_pop - num_parents

# Initialize population
initial_population = np.random.randint(2, size=pop_size).astype(int)
print(f'Population size = {pop_size}')
print(f'Initial population: \n{initial_population}')

# Fitness function
def cal_fitness(weight, value, population, threshold):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        total_value = np.sum(population[i] * value)
        total_weight = np.sum(population[i] * weight)
        fitness[i] = total_value if total_weight <= threshold else 0
    return fitness.astype(int)

# Selection function (corrected)
def selection(fitness, num_parents, population):
    parents = np.empty((num_parents, population.shape[1]))
    fitness = fitness.copy()  # Avoid modifying the original fitness array
    for i in range(num_parents):
        max_fitness_idx = np.argmax(fitness)
        parents[i, :] = population[max_fitness_idx, :]
        fitness[max_fitness_idx] = -999999  # Use large negative integer instead of -np.inf
    return parents

# Crossover function
def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = parents.shape[1] // 2
    crossover_rate = 0.8
    for i in range(num_offsprings):
        parent1_index = i % parents.shape[0]
        parent2_index = (i + 1) % parents.shape[0]
        if rd.random() <= crossover_rate:
            offsprings[i, :crossover_point] = parents[parent1_index, :crossover_point]
            offsprings[i, crossover_point:] = parents[parent2_index, crossover_point:]
        else:
            offsprings[i, :] = parents[parent1_index, :]  # No crossover, copy parent
    return offsprings

# Mutation function
def mutation(offsprings):
    mutation_rate = 0.4
    mutants = offsprings.copy()
    for i in range(mutants.shape[0]):
        if rd.random() <= mutation_rate:
            mutation_point = randint(0, mutants.shape[1] - 1)
            mutants[i, mutation_point] = 1 - mutants[i, mutation_point]  # Flip bit
    return mutants

# Optimization function
def optimize(weight, value, population, pop_size, num_generations, threshold):
    fitness_history = []
    num_parents = int(pop_size[0] / 2)
    num_offsprings = pop_size[0] - num_parents
    
    for _ in range(num_generations):
        # Calculate fitness
        fitness = cal_fitness(weight, value, population, threshold)
        fitness_history.append(fitness)
        
        # Select parents
        parents = selection(fitness, num_parents, population)
        
        # Generate offsprings via crossover
        offsprings = crossover(parents, num_offsprings)
        
        # Apply mutation
        mutants = mutation(offsprings)
        
        # Create new population
        population[0:num_parents, :] = parents
        population[num_parents:, :] = mutants
    
    # Final fitness calculation
    print(f'Last generation: \n{population}')
    fitness_last_gen = cal_fitness(weight, value, population, threshold)
    print(f'Fitness of the last generation: \n{fitness_last_gen}')
    
    # Select best solution
    max_fitness_idx = np.argmax(fitness_last_gen)
    best_solution = population[max_fitness_idx, :]
    
    return [best_solution], fitness_history

# Run optimization
parameters, fitness_history = optimize(weight, value, initial_population, pop_size, num_generations, knapsack_threshold)
print(f'The optimized parameters for the given inputs are: \n{parameters}')

# Display selected items
selected_items = item_number[parameters[0].astype(bool)]
print('\nSelected items that will maximize the knapsack without breaking it:')
for item in selected_items:
    print(item)

# Plot fitness history
fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
fitness_history_max = [np.max(fitness) for fitness in fitness_history]
plt.plot(range(num_generations), fitness_history_mean, label='Mean Fitness')
plt.plot(range(num_generations), fitness_history_max, label='Max Fitness')
plt.legend()
plt.title('Fitness through the Generations')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.show()

# Print fitness history shape
print(f'Fitness history shape: {np.asarray(fitness_history).shape}')