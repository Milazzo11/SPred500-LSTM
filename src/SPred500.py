"""
SPred500 stock time-series modeling system.

:author: Max Milazzo
"""


if __name__ == "__main__":
    print("LOADING SETUP...")
    # display initial setup loading message on program start


import os
import time
import pickle
import multiprocessing
import GA_funcs as GAF
from NN_prefs import load_prefs, schematize
from LSTM import LSTM_NN
from GA import GA
from GA_funcs import *
from tkinter import *
from tkinter import ttk


ICON_PATH = os.path.join("resources", "icon.ico")
# icon filepath


MODEL_DIR_PATH = "models"
# temporary folder to hold serialized models


CONFIRM_START_TIME = 1.5
# number of seconds to display model loading confirmation message in window
# before clearing


def q_train(model_queue, model, identifier):
    """
    Trains a new model, writes it to a temporary file, and adds the filename
    and model RMSE to multiprocessing queue.
    
    :param model_queue: multiprocessing queue to store data
    :param model: model instance to train
    :param identifier: unique model identifier
    
    :return: None; however, the pushing to the model_queue acts like a return
        variable -- and the value pushed is a tuple containing the model file
        the model's RMSE value
    """
    
    rmse = model.train()
    # train model
    
    model_file = os.path.join(MODEL_DIR_PATH, identifier + ".pkl")
    # get model filepath based on unique identifier
    
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
        # create serialized model file
    
    model_queue.put((model_file, rmse))
    # add filepath and RMSE to multiprocessing queue


def GA_multi_plots(model_queue, num_models, prefs, n_lookback, file_tag):
    """
    Generates multiple GA prediction plots based on trained models.
    
    :param model_queue: multiprocessing queue holding trained model access
        information and RMSE data  
    :param num_models: the number of models and plots to generate
    :param prefs: application preferences    
    :param n_lookback: number of days back to include in data for prediction
    :param file_tag: model generation identifier tag included in final output
        plot filename
    
    :return: best model trained (with highest RMSE), RMSE
    """
    
    print("\n----------\nGENERATING PREDICTION PLOTS\n")
    
    if n_lookback == 1:
        n_lookback_flag_text = "_NLOOKBACK=1"
        # set n_looKback=1 flag text
        
    else:
        n_lookback_flag_text = ""
        # set empty string for n_loopback=1 flag text
    
    best_rmse = 1
    # value to be set to lowest determined RMSE
    
    for i in range(num_models):
        model_file, cur_rmse = model_queue.get()
        # fetch all models and RMSE values from trained models
        
        with open(model_file, "rb") as f:
            cur_model = pickle.load(f)
            # load model
    
        if cur_rmse is None:
            return None, None
            # exit with error status
            
        cur_model.predict(
            show_plot=False, save_plot=True,
            filename=file_tag + str(i) + "_RMSE=" + str(cur_rmse) + 
            n_lookback_flag_text + ".png"
        )
        # save forecast charts for all models trained with optimized schema

        if cur_rmse <= best_rmse:
            best_rmse = cur_rmse
            best_model = cur_model
            # update best model
            
    return best_model, best_rmse
    

def GA_multi_preds(ticker, prefs, n_forecast, print_tag, file_tag, num_models,
        schema_list):
    """
    Generates multiple GA models and predictions.
    (note that the order of passed arguments in this function is very
    intentional such that arguments can easily be passed from the GA class)
    
    :param ticker: stock ticker
    :param prefs: application preferences
    :param n_forecast: number of days to make a prediction for
    :param print_tag: model generation identifier tag included in model display
        output message
    :param file_tag: model generation identifier tag included in final output
        plot filename
    :param num_models: the number of models and plots to generate
    :param schema_list: list of schemas to generate models with
    
    :return: model, RMSE
    """
    
    print("\n----------\nGENERATING " + print_tag + " MODELS\n")
    
    model_queue = multiprocessing.Queue()
    # initialize multiprocessing queue to hold model files and RMSE
    
    for i in range(num_models):
        n_lookback = int(schema_list[i]["LOOKBACK_MULT"] * n_forecast) + 1
        # get lookback associated with current schema
        
        cur_model = LSTM_NN(
            ticker, schema_list[i]["DATA_PERIOD"], n_lookback,
            n_forecast, schema_list[i], prefs["EVAL_PARTITION"],
            prefs["PRED_PLOT_FRAC"]
        )
        # initialize new model instance with finalized schema
        
        multiprocessing.Process(
            target=q_train, args=(
                model_queue, cur_model,
                ticker + str(n_forecast) + "D_" + file_tag + "_PID" + 
                str(multiprocessing.current_process().pid) + "." + str(i) +
                "_TIME" + str(time.time())
            )
        ).start()
        # start training process for each model instance
    
    return GA_multi_plots(model_queue, num_models, prefs, n_lookback, file_tag)
    # generate final GA predictions


def GA_gen(ticker, prefs, n_forecast):
    """
    Generates genetic algorithm optimized models and predictions.
    
    :param ticker: stock ticker
    :param prefs: application preferences
    :param n_forecast: number of days to make a prediction for
    
    :return: model, RMSE
    """
    
    print("STARTING GA OPTIMIZATION FOR " + ticker)

    optimizer = GA(
        ticker, prefs["POPULATION_SIZE"], prefs["MUTATION_RATE"],
        prefs["CROSSOVER_RATE"], prefs["ELITISM"], prefs["RANDOM_INITS"],
        schematize(prefs.copy()), prefs["DEF_MUTATION_RATE"]
    )
    # initialize new GA instance
    
    optimized_schema = optimizer.run(
    
        GAF.individual_generator, GAF.fitness_function,
        (
            n_forecast,
            prefs["DATA_PERIOD"],
            prefs["EVAL_PARTITION"]
        ),
        # fitness function argument tuple
        
        GAF.crossover_function, GAF.mutation_function, GAF.finalization_function,
        prefs["GA_FIN_PLOT"], GA_multi_preds,
        (
            ticker,
            prefs, 
            n_forecast,
            "GA SURVIVING POPULATION",
            "GA_TESTMODEL"
        ),
        # GA final display function argument tuple

        prefs["NUM_GENERATIONS"]
        # base params
        
    )
    # run GA
    
    if optimized_schema is None:
        return None, None
        # exit with error status

    model, rmse = GA_multi_preds(
        ticker, prefs, n_forecast, "FINAL", "GA_FINMODEL", prefs["FINAL_RUNS"],
        [optimized_schema] * prefs["FINAL_RUNS"]
    )
    # generate final models and predictions
    
    if model is None:
        return None, None
        # exit with error status
            
    print("OPTIMIZED SCHEMA: " + str(optimized_schema) + "\n")
    
    n_lookback = int(optimized_schema["LOOKBACK_MULT"] * n_forecast) + 1
    # calculate final model lookback
    
    if n_lookback == 1:
        print(f"FLAG ({ticker}, {n_forecast} day): n_lookback=1\n")
        # print n_lookback=1 indicator flag
        
    return model, rmse


def default_gen(ticker, prefs, n_forecast):
    """
    Generates default LSTM model and prediction.
    
    :param ticker: stock ticker
    :param prefs: application preferences
    :param n_forecast: number of days to make a prediction for
    
    :return: model, RMSE
    """
    
    n_lookback = int(prefs["LOOKBACK_MULT"] * n_forecast) + 1
    # calculate default lookback
        
    model = LSTM_NN(
        ticker, prefs["DATA_PERIOD"], n_lookback, n_forecast,
        schematize(prefs.copy()), prefs["EVAL_PARTITION"],
        prefs["PRED_PLOT_FRAC"]
    )
    # initialize model with default schema
    
    rmse = model.train()
    # train default model
    
    if rmse is None:
        return None, None
        # exit with error status
        
    return model, rmse


def generate_model(ticker, n_forecast, GA_optimize, show_plot):
    """
    Generate new LSTM model and create a prediction plot.
    
    :param ticker: stock ticker
    :param n_forecast: number of days to make a prediction for
    :param GA_optimize: whether or not to use GA optimization
    :param show_plot: whether plot should be displayed on-screen or just
        saved as a file
    """
    
    prefs = load_prefs()
    # get preferences

    if GA_optimize:
        model, rmse = GA_gen(ticker, prefs, n_forecast)
        # generate GA optimized model
        
    else:
        model, rmse = default_gen(ticker, prefs, n_forecast)
        # generate default model
        
    if model == None:
        print(ticker + " MODEL GENERATION FAILED")
        return
    
    print(ticker, n_forecast, "DAY FORECAST MODEL COMPLETE")
    print("FINAL TOP RMSE:", rmse)
    print("=========================\n")
    
    model.predict(
        show_plot=show_plot, save_plot=not GA_optimize,
        filename="DEFMODEL_RMSE=" + str(rmse) + ".png"
    )
    # make final prediction
    # (save file if not in GA optimization mode, since image would already be saved)
    
    print()


def start(window, loading_title, ticker_entry, period_entry, GA_optimize, show_plot):
    """
    Start model generation process.
    
    :param window: application window
    :param loading_title: text label to display "loading" information
    :param ticker_entry: ticker entry widget
    :param period_entry: forecast period entry widget
    :param GA_optimize: whether or not to use GA optimization
    :param show_plot: whether plot should be displayed on-screen or just
        saved as a file
    """
    
    tickers = ticker_entry.get()
    n_forecast = period_entry.get()
    # get entry values
    
    if tickers == "":
        print("ERROR - Enter Ticker(s)")
        return
    
    if not n_forecast.isnumeric():
        print("ERROR - Invalid Forecast Period")
        return

    tickers = tickers.replace(",", " ").split()
    # get all entered tickers
    
    for ticker in tickers:
        print(f"INITIATING NEW TRAINING PROCESS ({ticker})")
        
        multiprocessing.Process(
            target=generate_model, args=(
                ticker.strip(), int(n_forecast), bool(GA_optimize),
                bool(show_plot)
            )
        ).start()
        # generate a model for each stock ticker
    
    ticker_entry.delete(0, END)
    period_entry.delete(0, END)
    # clear entry contents
    
    loading_title.config(text="PROCESS INITIATED")
    window.after(
        int(CONFIRM_START_TIME * 1000), lambda: loading_title.config(text="")
    )
    # show loading confirmation message


def clear_models():
    """
    Clears models in model file from previous application use.
    """
    
    for file in os.listdir(MODEL_DIR_PATH):
        path = os.path.join(MODEL_DIR_PATH, file)
    
        if os.path.isfile(path):
            os.remove(path)


def main():
    """
    Launches application GUI.
    """
    
    clear_models()
    # clear old model files
    
    window = Tk()
    window.title("SPred500")
    window.iconbitmap(ICON_PATH)
    # create window

    title = l1 = Label(
        window, text="SPred500 stock time-series modeling system",
        font="arial 10 bold"
    )
    title.grid(row=0, column=0, columnspan=2, pady=10)
    # create title text
    
    title_break = ttk.Separator(window, orient=HORIZONTAL)
    title_break.place(x=0, y=30, relwidth=1)
    # create horizontal break line
    
    ticker_label = Label(window, text="Stock Ticker(s):", font="arial 9")
    ticker_label.grid(row=1, column=0, sticky=W, pady=2)
    # create ticker entry label
    
    period_label = Label(window, text="Forecast Period (days):", font="arial 9")
    period_label.grid(row=2, column=0, sticky=W, pady=2)
    # create forecast period entry label

    ticker_entry = Entry(window)
    ticker_entry.grid(row=1, column=1, pady=2)
    # create ticker entry box
    
    period_entry = Entry(window)
    period_entry.grid(row=2, column=1, pady=2)
    # create forecast period entry box
    
    GA_check_state = IntVar()
    GA_check = Checkbutton(
        window, text="Genetic Algorithm Optimization", variable=GA_check_state,
        onvalue=1, offvalue=0
    )
    GA_check.grid(row=3, column=0, columnspan=2, sticky=W, padx=30, pady=(20, 2))
    # create GA optimizer checkbox
    
    plot_check_state = IntVar()
    plot_check = Checkbutton(
        window, text="Display Forecast Plot(s)", variable=plot_check_state,
        onvalue=1, offvalue=0
    )
    plot_check.grid(row=4, column=0, columnspan=2, sticky=W, padx=30, pady=2)
    # create show plot checkbox
    
    loading_title = Label(window, text="", font="arial 9")
    loading_title.grid(row=5, column=1, pady=(20, 5), sticky=W)
    # create loading text label

    train_button = Button(window, text="Generate Model", 
        command=lambda: start(
            window, loading_title, ticker_entry, period_entry,
            GA_check_state.get(), plot_check_state.get()
        )
    )    
    train_button.grid(row=5, column=0, padx=5, pady=(20, 5), sticky=W)
    # create model generation initiation button
    
    print("SETUP COMPLETE\n")
    window.mainloop()


if __name__ == "__main__":
    main()