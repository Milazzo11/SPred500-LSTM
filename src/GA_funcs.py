"""
Defines functions used to run genetic algorithm optimization for LSTM schema.

:author: Max Milazzo
"""


import random
from LSTM import LSTM_NN


INDIVIDUAL_PARAMS = {

    # "key" : ((lower_gen_range, upper_gen_range), (lower_enforced_range, upper_enforeced_range))
    "LOOKBACK_MULT"  : ( (0, 2),      (0, 2) ),
    "DATA_PERIOD"    : ( (12, 120),   (1, float("inf")) ),
    
    "*LSTM_SCHEMA.0" : ( (10, 100),   (2, float("inf")) ),             # LSTM_SCHEMA, index 0
    "*LSTM_SCHEMA.1" : ( (-100, 100), (-float("inf"), float("inf")) ), # LSTM_SCHEMA, index 1
    "*LSTM_SCHEMA.2" : ( (-100, 100), (-float("inf"), float("inf")) ), # LSTM_SCHEMA, index 2
    "*LSTM_SCHEMA.3" : ( (-5, 5),     (-float("inf"), float("inf")) ), # LSTM_SCHEMA, index 3
    
    "*DROPOUT.0"     : ( (-0.3, 0.3), (-float("inf"), 0.9) ),          # DROPOUT, index 0
    "*DROPOUT.1"     : ( (-0.3, 0.3), (-float("inf"), 0.9) ),          # DROPOUT, index 1
    "*DROPOUT.2"     : ( (-0.3, 0.3), (-float("inf"), 0.9) ),          # DROPOUT, index 2
    "*DROPOUT.3"     : ( (-0.3, 0.3), (-float("inf"), 0.9) ),          # DROPOUT, index 3
    
    "PATIENCE"       : ( (1, 5),      (1, float("inf")) ),
    "*ATTENTION.0"   : ( (-1, 1),     (-float("inf"), float("inf")) ),
    "*ATTENTION.1"   : ( (-1, 1),     (-float("inf"), float("inf")) ),
    
    "BATCH_SIZE"     : ( (1, 32),     (1, float("inf")) ),
    "EPOCHS"         : ( (1, 100),    (1, float("inf")) )
    
}
# defines individual random generation ranges and legal (returns non-zero fitness) ranges


INDIVIDUAL_FUNCS = {

    "LOOKBACK_MULT"  : lambda x: x,
    "DATA_PERIOD"    : lambda x: int(x),
    
    "*LSTM_SCHEMA.0" : lambda x: int(x),
    "*LSTM_SCHEMA.1" : lambda x: int(x) if x > 1 else None,
    "*LSTM_SCHEMA.2" : lambda x: int(x) if x > 1 else None,
    "*LSTM_SCHEMA.3" : lambda x: x if x > 0 else None,
    
    "*DROPOUT.0"     : lambda x: x if x > 0 else None,
    "*DROPOUT.1"     : lambda x: x if x > 0 else None,
    "*DROPOUT.2"     : lambda x: x if x > 0 else None,
    "*DROPOUT.3"     : lambda x: x if x > 0 else None,
    
    "PATIENCE"       : lambda x: int(x),
    "*ATTENTION.0"   : lambda x: True if x > 0 else False,
    "*ATTENTION.1"   : lambda x: True if x > 0 else False,
    
    "BATCH_SIZE"     : lambda x: int(x),
    "EPOCHS"         : lambda x: int(x)

}
# defines conversion functions to map genes to final hyper-parameter settings
# used by the LSTM model


CROSSOVER_COMBINATION_CHANCE = 0.5
# defines the chance of gene combination when performing crossover
# (as opposed to just selecting one gene or the other from a parent)


def finalization_function(individual):
    """
    Converts individual in list format to a correctly scaled preference dictionary
    (generates the finalized form of an individual to be used by LSTM).
    
    :param individual: individual in list format
    :return: preference dictionary
    """

    prefs = {}
    individual_index = 0
    
    multivalue_prefs = []
    multivalue_key = ""
    
    for key in INDIVIDUAL_PARAMS:        
        if len(multivalue_prefs) != 0 and "*" + multivalue_key != key.split(".")[0]:
            prefs[multivalue_key] = multivalue_prefs
            multivalue_prefs = []
            # copy temporary list of values to multivalue attribute
        
        if key[0] == "*":
            multivalue_key = key[1:].split(".")[0]
            multivalue_prefs.append(
                INDIVIDUAL_FUNCS[key](individual[individual_index])
            )
            # add multivalue attribute to temporary list
            
        else:
            prefs[key] = INDIVIDUAL_FUNCS[key](individual[individual_index])
            # add singleton attribute to dictionary
            
        individual_index += 1
        
    return prefs
    
    
def definalization_num_convert(val):
    """
    Maps non-numeric values in finalized dictionary to numeric representations.
    
    :param val: value being observed
    :return: numeric representation of value
    """
    
    if isinstance(val, bool):
        if val:
            return 1
            # True -> 1
            
        else:
            return 0
            # False -> 0
    
    if isinstance(val, int) or isinstance(val, float):
        return val
        # integers and floats do not change value
    
    if val is None:
        return 0
        # None -> 0
    
    return 0
    # unexpected value encountered; setting to 0
    
    
def definalization_function(individual_dict):
    """
    Converts individual in finalized dictionary format to list format that is
    GA optimizable.
    
    :param individual_dict: individual in dictionary format
    :return: individual in list format
    """
    
    definalized_individual = []
    
    for key in INDIVIDUAL_PARAMS:
        if key[0] == "*":
            cur_key, index = key[1:].split(".")
        
            definalized_individual.append(
                definalization_num_convert(individual_dict[cur_key][int(index)])
            )
            # add all parts of multivalue attributes to list
            
        else:
            definalized_individual.append(
                definalization_num_convert(individual_dict[key])
            )
            # add singleton values to list
            
    return definalized_individual


def mutation_function(gene):
    """
    Mutates a single gene.
    
    :param gene: unmutated gene
    :return: mutated gene
    """
    
    return gene + gene * random.uniform(-1, 1) ** 2
    # mutated gene is scaled based on unmutated value


def rand_individual_generator():
    """
    Generate a new constrained random individual.
    
    :return: new generated individual
    """
    
    individual = []
    
    for key in INDIVIDUAL_PARAMS:
        individual.append(
            random.uniform(
                INDIVIDUAL_PARAMS[key][0][0], INDIVIDUAL_PARAMS[key][0][1]
            )
        )
        # generate each constrained random gene in individual
        
    return individual    
    

def individual_generator(mode, default=None, mutation=None, mutator=None):
    """
    Generate a new individual based on defined standards.
    
    :param mode: individual generation mode
        0 - constrained random gene generation
        1 - default indivdual
        2 - n_lookback = 1, special case indidivual
        3 - mutated default individual
    :param default: a default individual in dictionary format
    :param mutation: the mutation rate used when generating
        the mutated versions of the default individul
    :param mutator: individual mutator function

    :return: new generated individual
    """
    
    if mode == 0:
        return rand_individual_generator()
        # generate constrained random individual
        
    else:
        if mode == 2:
            default["LOOKBACK_MULT"] = 0
            # force n_lookback=1

        individual = definalization_function(default)
        # turn default individual dictionary into optimizable list
        
        if mode == 3:
            individual = mutator(individual, mutation, mutation_function)
            # generate mutated default individual
            
        return individual
            

def crossover_function(parent1, parent2):
    """
    Performs crossover between two parents to create two new children.
    
    :param parent1: parent #1
    :param parent2: parent #2
    
    :return: new offspring (child #1, child #2)
    """
    
    children = [
        [], # child 1
        []  # child 2
    ]
    
    for child_index in range(2):
        for gene_index in range(len(parent1)):
            if random.random() > CROSSOVER_COMBINATION_CHANCE:
                children[child_index].append(
                    0.5 * parent1[gene_index] + 0.5 * parent2[gene_index]
                )
                # new child gene is combined average of parent genes
                
            elif random.random() > 0.5:
                children[child_index].append(parent1[gene_index])
                # select parent 1 gene
                
            else:
                children[child_index].append(parent2[gene_index])
                # select parent 2 gene
                
    return children[0], children[1]
    
    
def has_legal_ranges(individual):
    """
    Ensures that an individual's genes are within legal ranges.
    
    :param individual: the individual being checked
    :return: whether or not the individual's ranges are legal
    """
    
    index = 0
    
    for key in INDIVIDUAL_PARAMS:
        lower_bound = INDIVIDUAL_PARAMS[key][1][0]
        upper_bound = INDIVIDUAL_PARAMS[key][1][1]
        # get lower and upper bounds for gene
        
        if individual[index] < lower_bound or individual[index] > upper_bound:
            return False
            
        index += 1
            
    return True


def fitness_function(ticker, fitness_queue, individual, individual_index,
        n_forecast, data_period, eval_partition):
    """
    Runs LSTM model to determine individual fitness.
    
    :param ticker: stock ticker
    :param fitness_queue: multiprocessing queue to store final fitness result
    :param individual: individual in genetic algorithm population
    :param individual_index: index of the individual being evaluated
    :param n_forecast: number of days to make a prediction for
    :param data_period: period of time from which to fetch stock data
    :param eval_partition: fraction of the training data to use for model

    :return: None; however, the pushing to the fitness_queue acts like a return
        variable -- and the value pushed is a tuple containing the individual's 
        fitness value and index (1 / RMSE, individual_index)
    """
    
    if not has_legal_ranges(individual):
        fitness_queue.put((0, individual_index))
        # handles individual with an invalid range
        
        print("[ REMOVED INDIVIDUAL WITH INVALID RANGE ]")
    
    schema = finalization_function(individual)
    # finalize individual to get LSTM schema

    n_lookback = int(schema["LOOKBACK_MULT"] * n_forecast) + 1
    # calculate lookback
      
    model = LSTM_NN(
        ticker, data_period, n_lookback, n_forecast, schema, eval_partition
    )
    # initialize new model with GA schema
        
    rmse = model.train()
    # train the model initialized with GA schema and get RMSE
    
    if rmse is None:
        fitness_queue.put((None, None))
        # handles download error case
    
    elif rmse == 0:
        fitness_queue.put((float("inf"), individual_index))
        # handles very unlikely case of 0 RMSE ("perfect" model)
        
    else:
        fitness_queue.put((1 / rmse, individual_index))
        # inverts RMSE so lower values have higher fitness