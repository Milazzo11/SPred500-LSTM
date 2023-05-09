"""
GA (genetic algorithm) implementation.

:author: Max Milazzo
"""


import random
import multiprocessing
import numpy as np


class GA:
    """
    GA (genetic algorithm) object definition.
    """
    
    def __init__(self, label, population_size, mutation_rate, crossover_rate,
            elitism, initial_random_frac, default_individual,
            default_mutation_rate):
        """
        GA initialization.
        
        :param label: a display label included in a print statement for each
            generation -- should provide additional user information
        :param population_size: size of the population
        :param mutation rate: rate of gene mutation
        :param crossover_rate: rate of crossover (as opposed to parents
            surviving to the next generation)
        :param elitism: should GA instance employ elitism
        :param initial_random_frac: the fraction of the initial GA population
            that is randomly generated; the rest is created based on the
            default prefernces and alterations of these preferences, as well as
            at least one special case
        :param default_individual: the "default" individual included in initial
            population generation (not necessarily in compatible format)
        :param default_mutation_rate: the mutation rate used when generating
            the mutated versions of the default individul created during
            initial population generation
        """
        
        self.label = label
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.initial_random_frac = initial_random_frac
        self.default_individual = default_individual
        self.default_mutation_rate = default_mutation_rate
        self.best_individual = None
        self.best_fitness = None


    def _initialize_population(self, individual_generator):
        """
        Initialize the population.
        
        :param individual_generator: function to generate an individual
        """
        
        random_count = int(self.population_size * self.initial_random_frac)
        mutduf_count = self.population_size - random_count - 2
        # define the number of fully random and mutated default individuals
        # (2 is subtracted from the mutated default individual count because
        # one insidivual will be the default, and another will is reserved for
        # a special case)
        
        self.population = [
            individual_generator(mode=0) for _ in range(random_count)
        ]
        # generate random individuals (mode=0 "random")
        
        if self.population_size - random_count > 0:
            self.population.append(
                individual_generator(
                    mode=1, default=self.default_individual.copy()
                )
            )
            # add the finalized default individual if there is enough space in
            # the population to do so
            
        if self.population_size - random_count > 1:
            self.population.append(
                individual_generator(
                    mode=2, default=self.default_individual.copy()
                )
            )
            # add special case individual if there is enough space in the
            # population to do so (mode=1 "special case")
        
        if mutduf_count > 0:
            self.population.extend([
                individual_generator(
                    mode=3, default=self.default_individual.copy(),
                    mutation=self.default_mutation_rate,
                    mutator=self.mutate
                ) for _ in range(mutduf_count)
            ])
            # generate mutated default individuals if there is enough space
            # in the population to do so (mode=2 "mutated default")


    def _evaluate_fitness(self, fitness_function, fitness_calc_arg_tuple):
        """
        Evaluates the fitness of individuals in the population.
        
        :param fitness_function: function to test genes and determine fitness
        :param fitness_calc_arg_tuple: tuple containing additional arguments
            the fitness_function may need to properly function
            
        :return: fitness values
        """
        
        fitness_queue = multiprocessing.Queue()
        # multiprocessing queue to fetch returned fitness values
        
        for i in range(self.population_size):
            process = multiprocessing.Process(
                target=fitness_function, args=(
                    self.label, fitness_queue, self.population[i], i,
                    *fitness_calc_arg_tuple
                )
            ).start()
            # create and start new process for all members of the population
        
        fitnesses = [0] * self.population_size
        # initialize "empty" list to store fitness values
        
        for _ in range(self.population_size):
            fitness, index = fitness_queue.get()
            
            if fitness is None:
                return None
                # handles error return state
                
            fitnesses[index] = fitness
            # set fitness values in correct place in list
        
        return fitnesses


    def _select_parents(self, fitnesses):
        """
        Selects parents for crossover based on their fitness.
        
        :param fitnesses: fitness values
        :return: selected parents
        """
        
        parents = []
        
        for _ in range(2):
            candidates = random.choices(self.population, weights=fitnesses, k=2)
            candidates_fitnesses = [
                fitnesses[
                    self.population.index(candidate)
                ] for candidate in candidates
            ]
            # get canidates and canidate fitness values
            
            selected_index = np.argmax(candidates_fitnesses)
            parents.append(candidates[selected_index])
            # add parents to return list
            
        return parents


    def mutate(self, individual, mutation_rate, mutation_function):
        """
        Performs mutations on an individual.
        
        :param individual: individual being mutated
        :param mutation_function: function performing gene mutation
        
        :return: mutated individual
        """
        
        mutated_individual = []
        
        for gene in individual:
            if random.random() < mutation_rate:
                mutated_gene = mutation_function(gene)
                mutated_individual.append(mutated_gene)
                # mutate gene
                
            else:
                mutated_individual.append(gene)
                # leave gene unmutated
                
        return mutated_individual


    def _crossover(self, parent1, parent2, crossover_function):
        """
        Performs crossover.
        
        :param parent1: parent #1
        :param parent2: parent #2
        :param crossover_function: function performing gene crossover
        
        :return: child1, child2
        """
        
        if random.random() < self.crossover_rate:
            return crossover_function(parent1, parent2)
            # return children of crossover 

        else:
            return parent1, parent2
            # parents continue to next generation instead of children

    
    def _fin_display(self, fin_output_function, fin_output_arg_tuple, finalization_function):
        """
        Executes output function for finalization of all individuals.
        
        :param fin_output_function: output function for final generation schema
        :param fin_output_arg_tuple: tuple containing additional arguments
            the fin_output_function may need to properly function
        :param finalization_function: function to finalize an individual's format
        """
        
        schema_list = [
            finalization_function(individual)
            for individual in self.population
        ]
        # generate schema list for final generation
        
        fin_output_function(
            *fin_output_arg_tuple, self.population_size, schema_list
        )
        # call display function on last generation
    
    
    def _gen_next_population(self, best_individual, fitnesses,
            crossover_function, mutation_function):
        """
        Generates the next generation of individuals.
        
        :param best_individual: best individual from the previous generation
        :param fitnesses: fitness values
        :param crossover_function: function performing gene crossover
        :param mutation_function: function performing gene mutation
        
        :return: the population for the next generation
        """
        
        next_population = []

        if self.elitism:
            next_population.append(best_individual)
            # add best individual to the next generation if elitism enabled

        while len(next_population) < self.population_size:
            parent1, parent2 = self._select_parents(fitnesses)
            child1, child2 = self._crossover(
                parent1, parent2, crossover_function
            )
            # select and breeds parents
            
            child1 = self.mutate(
                child1, self.mutation_rate, mutation_function
            )
            child2 = self.mutate(
                child2, self.mutation_rate, mutation_function
            )
            # mutate children
            
            next_population.append(child1)
            # adds child #1 to next generation
            
            if len(next_population) < self.population_size:
                next_population.append(child2)
                # add child #1 to the next generation if the population size is
                # not exceeded
                
        return next_population
    
    
    def run(self, individual_generator, fitness_function, fitness_calc_arg_tuple,
            crossover_function, mutation_function, finalization_function, 
            fin_output, fin_output_function, fin_output_arg_tuple, num_generations):
        """
        Runs the genetic algorithm.
        
        :param individual_generator: function to generate an individual
        :param fitness_function: function to test genes and determine fitness
        :param fitness_calc_arg_tuple: tuple containing additional arguments
            the fitness_function may need to properly function
        :param crossover_function: function performing gene crossover
        :param mutation_function: function performing gene mutation
        :param finalization_function: function to finalize an individual's format
        :param fin_output: whether or not to call output function on last generation
        :param fin_output_function: output function for final generation schema
        :param fin_output_arg_tuple: tuple containing additional arguments
            the fin_output_function may need to properly function
        :param num_generations: number of generations to run the algorithm for
        
        :return: finalized best individual
        """
        
        self._initialize_population(individual_generator)
        # initialize the population

        for generation in range(num_generations):
            print(f"\n-----\nGA GENERATION #{generation + 1} for {self.label}\n")

            fitnesses = self._evaluate_fitness(
                fitness_function, fitness_calc_arg_tuple
            )
            # get population fitness values
            
            if fitnesses is None:
                return None
                # handle error return state

            best_index = np.argmax(fitnesses)
            best_individual = self.population[best_index]
            best_fitness = fitnesses[best_index]
            # determine best individual and fitness

            if self.best_individual is None or best_fitness > self.best_fitness:
                self.best_individual = best_individual
                self.best_fitness = best_fitness
                # update GA instance's best individual and fitness

            self.population = self._gen_next_population(
                best_individual, fitnesses, crossover_function,
                mutation_function
            )
            # advance to the next generation
            
        if fin_output:
            self._fin_display(
                fin_output_function, fin_output_arg_tuple,
                finalization_function
            )
            # call display function on last generation

        return finalization_function(self.best_individual)