"""
Loads default application preferences.

:author: Max Milazzo
"""


import os


PREFS_PATH = os.path.join("resources", "PREFS.txt")
# preferences filepath


NON_LSTM_SCHEMA = (

    "NUM_GENERATIONS",
    "POPULATION_SIZE",
    "CROSSOVER_RATE",
    "MUTATION_RATE",
    "ELITISM",
    "RANDOM_INITS",
    "DEF_MUTATION_RATE",
    "TEST_RUNS",
    "FINAL_RUNS",
    "GA_FIN_PLOT",
    "EVAL_PARTITION",
    "PRED_PLOT_FRAC"
    
)
# defines all non-LSTM (non-optimizable) preferences/schema values


def is_float(string):
    """
    Determines whether a given string can be represented as a float.
    
    :param string: input string
    :return: True if it can, False otherwise
    """
    
    if string.replace(".", "").isnumeric():
        return True
    else:
        return False


def singleton_parse(pref_str):
    """
    Parses a singleton string preference value.
    
    :param pref_str: preference string
    :return: preference value
    """
    
    if pref_str.isnumeric():
        return int(pref_str)
        # handles int numeric preferences
    
    if is_float(pref_str):
        return float(pref_str)
        # handles non-int numeric preferences
    
    if pref_str.lower() == "true":
        return True
        # handles boolean preferences
        
    if pref_str.lower() == "false":
        return False
        # handles boolean preferences
        
    if pref_str.lower() == "none":
        return None
        # handles NoneType preferences
        
    return ""
    # not a singleton value


def parse_pref(pref_str):
    """
    Parses a preference string line.
    
    :param pref_str: preference string
    :return: preference key, preference
    """
    
    data = pref_str.split("=")
    pref = singleton_parse(data[1])
    
    if pref != "":
        return data[0], pref
        # singleton preference
        
    multi_pref_list = data[1].replace(",", " ").split()
    
    for index in range(len(multi_pref_list)):
        multi_pref_list[index] = singleton_parse(multi_pref_list[index])
        # get preference value of all items in list
        
    return data[0], multi_pref_list
        
    
def load_prefs():
    """
    Loads default application preferences.
    
    :return: preferences
    """
    
    prefs = {}
    
    with open(PREFS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            
            if len(line) == 0:
                continue
                # skip empty lines
            
            if line[0] != "#":
                pref_key, pref = parse_pref(line)
                prefs[pref_key] = pref
                # update preferences (on lines without comment symbol)
    
    return prefs
    
    
def schematize(prefs):
    """
    Remove non-LSTM (non-optimizable) schema from a preferences dictionary.
    
    :param prefs: preferences dictionary
    :return: schematized dictionary containing only LSTM schema
    """

    for nls in NON_LSTM_SCHEMA:
        prefs.pop(nls)
        # remove non-LSTM schema preferences
        
    return prefs