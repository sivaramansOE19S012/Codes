"""
Created on Sun Dec  6 21:32:04 2020

@author: Sivaraman Sivaraj, Suresh Rajendran
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from tqdm import tqdm


def multiple_linear_regression(X,Y):
    """
    Parameters
    ----------
    X : Input Vectors
    Y : Output 
    Returns
    -------
    Dictionary: 
    {'COD': COD, 'coeff': coeff, 'error': av_error}

    """
    X_t = X.transpose()
    coeff = np.dot((np.linalg.pinv((np.dot(X_t, X)))), (np.dot(X_t, Y))) #here, we use pseudo inverse, instead of normal inverse
    Y_p = np.dot(X, coeff)
    Y_mean = np.mean(Y)
    SST = np.array([(i - Y_mean) ** 2 for i in Y]).sum()
    SSR = np.array([(i - j) ** 2 for i, j in zip(Y, Y_p)]).sum()
    COD = (1 - (SSR / SST)) * 100.0
    av_error = (SSR / len(Y))
    return {'COD': COD, 'coeff': coeff, 'error': av_error}

def check_termination_condition(best_individual,generation_count,max_generations):
    if ((best_individual['COD'] >= 80.0) or (generation_count == max_generations)):
        return True
    else:
        return False #we can declare the error percentage here
    
def create_individual(individual_size):
    return [np.random.rand() for i in range(individual_size)] #to have intial weight

def create_population(individual_size, population_size):
    return [create_individual(individual_size) for i in range(population_size)]


def get_fitness(individual, inputs,outputs):
    predicted_outputs = np.dot(np.array(inputs), np.array(individual))
    output_mean = np.mean(outputs)
    SST = np.array([(i - output_mean) ** 2 for i in outputs]).sum()
    SSR = np.array([(i - j) ** 2 for i, j in zip(outputs, predicted_outputs)]).sum()
    COD = (1 - (SSR / SST)) * 100.0
    average_error = (SSR / len(outputs))
    return {'COD': COD, 'error': average_error, 'coeff': individual}

def evaluate_population(population,selection_size,best_individuals_stash,inputs, outputs):
    fitness_list = [get_fitness(individual, inputs,outputs)for individual in tqdm(population)]
    error_list = sorted(fitness_list, key=lambda i: i['error'])
    best_individuals = error_list[: selection_size]
    best_individuals_stash.append(best_individuals[0]['coeff'])
    print('Error: ', best_individuals[0]['error'],
           'COD: ', best_individuals[0]['COD'])
    return best_individuals


def crossover(parent_1, parent_2,individual_size):
    child = {}
    loci = [i for i in range(0, individual_size)]
    loci_1 = np.random.choice(loci, int(np.floor(0.5*(individual_size))))
    loci_2 = [i for i in loci if i not in loci_1]
    chromosome_1 = [[i, parent_1['coeff'][i]] for i in loci_1]
    chromosome_2 = [[i, parent_2['coeff'][i]] for i in loci_2]
    child.update({key: value for (key, value) in chromosome_1})
    child.update({key: value for (key, value) in chromosome_2})
    return [child[i] for i in loci]

def mutate(individual,individual_size,probability_of_gene_mutating):
    loci = [i for i in range(0, individual_size)]
    no_of_genes_mutated = np.floor(probability_of_gene_mutating*individual_size)
    loci_to_mutate = np.random.choice(loci, int(no_of_genes_mutated))
    for locus in loci_to_mutate:
        gene_transform = np.random.choice([-1, 1])
        change = gene_transform*np.random.rand()
        individual[locus] = individual[locus] + change
    return individual


def get_new_generation(selected_individuals,population_size,individual_size,
                       probability_of_individual_mutating,probability_of_gene_mutating):
    parent_pairs = [np.random.choice(selected_individuals, 2)
                    for i in range(population_size)]
    offspring = [crossover(pair[0], pair[1],individual_size) for pair in parent_pairs]
    offspring_indices = [i for i in range(population_size)]
    offspring_to_mutate = np.random.choice(
        offspring_indices,
        int(np.floor(probability_of_individual_mutating*population_size))
    )
    mutated_offspring = [[i, mutate(offspring[i],individual_size,probability_of_gene_mutating)]
                         for i in offspring_to_mutate]
    for child in mutated_offspring:
        offspring[child[0]] = child[1]
    return offspring


def activate(X,Y,population_size,selection_size,max_generations,
             probability_of_individual_mutating,probability_of_gene_mutating):
    """
    Parameters
    ----------
    X : Input Vectors
    Y : Output Vectors
    population_size : Number of Input Vector
    selection_size : declare the number of chromosomes in batch
    max_generations : Declare the iteration length
    probability_of_individual_mutating : Declare b/w : 0-1
    probability_of_gene_mutating : Declare b/w : 0-1

    Returns
    -------
    Co-efficient of Determination, error, coefficients

    """
    inputs, outputs = X,Y
    individual_size = len(inputs[0])
    population_size = 100
    best_possible = multiple_linear_regression(inputs, outputs)
    best_individuals_stash = [create_individual(individual_size)] #we are generally intiating by random 
    initial_population = create_population(individual_size, population_size)
    current_population = initial_population
    termination = False
    generation_count = 0
    while termination is False:
        current_best_individual = get_fitness(best_individuals_stash[-1], inputs,outputs)
        print('Generation: ', generation_count)
        best_individuals = evaluate_population(current_population,
                                               selection_size,best_individuals_stash,X,Y)
        current_population = get_new_generation(best_individuals,population_size,individual_size,
                                                probability_of_individual_mutating,probability_of_gene_mutating)
        termination = check_termination_condition(current_best_individual,
                                                  generation_count,max_generations)
        generation_count += 1
        
    else:
        print("The final one:",get_fitness(best_individuals_stash[-1], inputs,outputs))
        ans = get_fitness(best_individuals_stash[-1], inputs,outputs)
        return ans['COD'],ans['error'],ans['coeff']
    



######################
### Testing ##########
######################
# data  = pd.read_csv('D.csv', header = 0)
# _X = data.iloc[:,0].tolist()
# _Y = data.iloc[:,1].tolist()
# X_i = list()
# for i in range(len(_X)):
#     temp = [_X[i],1]
#     X_i.append(temp)
# X = np.array(X_i)
# Y = np.array(_Y)
# a,b,c = activate(X,Y,len(X),10,20,0.25,0.1)
# print(c)
########################
########################







