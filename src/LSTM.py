"""
Stock preciction LSTM model.

:author: Max Milazzo
"""


import os
import sys
import math
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from Attention import Attention
from tensorflow.keras.layers import LSTM, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


PLOT_DIR = "plots"
# plot save directory


class LSTM_NN:
    """
    Stock prediction LSTM model object definition.
    """
    
    def __init__(self, ticker, data_period, n_lookback, n_forecast, schema,
            eval_partition, pred_plot_frac=None):
        """
        Initializes a new LSTM instance.

        :param ticker: stock ticker
        :param data_period: period of time from which to fetch stock data
        :param n_lookback: number of days back to include in data for prediction
        :param n_forecast: number of days to make a prediction for
        :param schema: dictionary holding data to build LSTM model schema
        :param eval_partition: fraction of the training data to use for model 
            accuracy evaluation
        :param pred_plot_frac: the fraction of the plot display consisting of
            the prediction line
        """
        
        self.ticker = ticker
        self.data_period = data_period
        self.n_lookback = n_lookback
        self.n_forecast = n_forecast
        self.schema = schema
        self.eval_partition = eval_partition
        self.pred_plot_frac = pred_plot_frac
        # set passed object variables
        
        self.raw_data, self.scaled_data, self.scaler = self._fetch_data()
        # fetch online data
        
    
    def _fetch_data(self):
        """
        Fetches stock data from Yahoo Finance and scales it.

        :return: raw stock data, scaled stock data, data scaler value
        """
        
        print("[ FETCHING " + self.ticker + " DATA ]")
        
        raw_data = yf.download(
            tickers=[self.ticker], period=str(self.data_period) + "mo",
            progress=False
        )
        # download the data
        
        if len(raw_data) == 0:
            print("\nERROR: DATA DOWNLOAD FAILURE")
            
            return [], None, None
            # exit function (error handled during training)
        
        raw_data = raw_data["Close"].fillna(method="ffill")
        scaled_data = raw_data.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(scaled_data)
        scaled_data = scaler.transform(scaled_data)
        # format and scale the data
        
        return raw_data, scaled_data, scaler
        

    def _gen_train_seqs(self):
        """
        Generates and sets training input and output sequences.
        """
        
        self.x_train = []
        self.y_train = []

        for i in range(self.n_lookback, len(self.scaled_data) - self.n_forecast + 1):
            self.x_train.append(self.scaled_data[i - self.n_lookback:i])
            self.y_train.append(self.scaled_data[i:i + self.n_forecast])
            # generate time-series data

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        # convert to numpy arrays
        
        
    def _test_train_split(self):
        """
        Partitions data into "training" and "testing" sets for model evaluation.

        :return: input (x) training sequence data, output (y) training sequence data,
            input (x) evaluation sequence data, output (y) evaluation sequence data
        """
        
        split_index = int(len(self.x_train) * (1 - self.eval_partition))
        
        return (self.x_train[:split_index], self.y_train[:split_index],
            self.x_train[split_index:], self.y_train[split_index:])
    
    
    def _config_LSTM_layers(self):
        """
        Configures model LSTM layers.
        """
        
        self.model.add(
            LSTM(
                units=self.schema["LSTM_SCHEMA"][0], return_sequences=True,
                input_shape=(self.n_lookback, 1)
            )
        )
        # add initial LSTM layer
        
        if self.schema["DROPOUT"][0] is not None:
            self.model.add(Dropout(self.schema["DROPOUT"][0]))
            # add optional DROPOUT layer preceeding inital LSTM layer
            
        if ((self.schema["LSTM_SCHEMA"][1] is not None or
            self.schema["LSTM_SCHEMA"][2] is not None) and
            self.schema["ATTENTION"][0]):
            # if at there is a preceeding LSTM layer and "attention" is active
            
            self.model.add(Attention(return_sequences=True))
            # add optional attention mechanism layer between LSTM layers
        
        for i in range(1, 3):
        # add deep hidden LSTM_SCHEMA and DROPOUT layers
        # (DROPOUT layer ignored if LSTM layer not added)
        
            if self.schema["LSTM_SCHEMA"][i] != None:
                self.model.add(
                    LSTM(
                        units=self.schema["LSTM_SCHEMA"][i],
                        return_sequences=True
                    )
                )
                # add LSTM layer
                
                if self.schema["DROPOUT"][i] != None:
                    self.model.add(Dropout(self.schema["DROPOUT"][i]))
                    # add DROPOUT layer


    def _config_output_layers(self):
        """
        Configures model Dense and output layers.
        """
        
        if self.schema["ATTENTION"][1]:
            self.model.add(Attention(return_sequences=False))
            # add optional attention mechanism layer before Dense layer(s)
        
        if self.schema["LSTM_SCHEMA"][3] is not None:
            self.model.add(
                Dense(int(self.n_forecast * self.schema["LSTM_SCHEMA"][3]) + 2)
            )
            # add DENSE layer
            
            if self.schema["DROPOUT"][3] is not None:  
                self.model.add(Dropout(self.schema["DROPOUT"][3]))
                # add DROPOUT layer
        
        self.model.add(Flatten())
        self.model.add(Dense(self.n_forecast))
        # add final FLATTEN and OUTPUT layer   
    
    
    def _fit_model(self):
        """
        Initializes and trains LSTM model.
        
        :return: model RMSE
        """

        self.model = Sequential()
        # create model
        
        self._config_LSTM_layers()
        self._config_output_layers()
        # configure model structure
        
        early_stop = EarlyStopping(
            monitor="val_loss", mode="min", patience=self.schema["PATIENCE"]
        )
        # create early stopping callback
        
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        # compile model
        
        history = self.model.fit(
            self.x_train, self.y_train, batch_size=self.schema["BATCH_SIZE"],
            epochs=self.schema["EPOCHS"],
            validation_data=(self.x_test, self.y_test), callbacks=[early_stop],
            verbose=0
        )
        # fit model
        
        rmse = math.sqrt(history.history["val_loss"][-1])
        # calculate RMSE based on validation loss
        
        print(
            "\nNEW " + self.ticker + " MODEL TRAINED\nRMSE: " + str(rmse) +
            "\nSCHEMA: " + str(self.schema) + "\n====================\n"
        )

        return rmse


    def _predict_scaled(self):
        """
        Generates and sets a scaled LSTM prediction.
        """
        
        pred_data = self.scaled_data[-self.n_lookback:]
        pred_data = pred_data.reshape(1, self.n_lookback, 1)
        # fetch last available input sequence and reshape

        self.scaled_forecast = self.model.predict(
            pred_data, verbose=0
        ).reshape(-1, 1)
        # make scaled prediction
    
    
    def _plot_data_config(self):
        """
        Configures and fetches data used in final forecast plot.
        """
        
        unscaled_forecast = self.scaler.inverse_transform(self.scaled_forecast)
        # descale forecast data

        date = pd.DataFrame(columns=["date"])
        date["date"] = self.raw_data.index.copy()
        cur_date = date["date"][len(date) - 1].to_pydatetime()
        # fetch current date (most recent date in raw price data)
        
        title = (
            str(cur_date.date()) + " " + self.ticker + " " + 
            str(self.n_forecast) + " Day Forecast"
        )
        # generate plot title
        
        pred_dates = [cur_date]
        pred_prices = np.append(self.raw_data[-1], unscaled_forecast)
        # add current date and price as starting point for forecast plot

        for _ in range(self.n_forecast):
            cur_date = cur_date + datetime.timedelta(days=1)
            pred_dates.append(cur_date)
            # construct list of future dates for prediction plot x-values
        
        return title, pred_dates, pred_prices


    def _show_forecast(self, show_plot, save_plot, filename):
        """
        Displays and saves final forecast plot.

        :param show_plot: whether plot should be displayed on-screen or just
            saved as a file
        :param save_plot: whether plot image should be saved to the disk
        :param filename: filename to give to saved plot (if plot is saved)
        """
        
        title, pred_dates, pred_prices = self._plot_data_config()

        plt.figure(figsize=(16, 8), num=title)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Close Price ($)")
        # configure plot size and labelling
        
        plt.plot(
            self.raw_data[-math.ceil(
                self.n_forecast / self.pred_plot_frac
            ):]
        )
        plt.plot(pred_dates, pred_prices)
        # plot data
        
        plt.legend(
            ["Historical Data", str(self.n_forecast) + " Day Forecast"],
            loc="upper left"
        )
        # create plot legend
        
        if save_plot:
            save_dir = os.path.join(PLOT_DIR, title)
            
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                # create save directory if it doesn't exist
            
            plt.savefig(os.path.join(save_dir, filename), bbox_inches="tight")
            # save plot as PNG
        
        if show_plot:
            plt.show()
            # halt execution and display plot
            
        plt.clf()
        # clear figure after use


    def train(self):
        """
        Trains a model using a subset of the available data, and then uses the rest
            to perform a model evaluation.
     
        :return: model RMSE (or None if no viable data)
        """
        
        if len(self.raw_data) == 0:
            return None
            # no viable data
        
        self._gen_train_seqs()
        self.x_train, self.y_train, self.x_test, self.y_test = self._test_train_split()
        # split training and testing data if in evaluation mode
        
        return self._fit_model()
        # fit model and return RMSE


    def predict(self, show_plot, save_plot, filename):
        """
        Generates a final prediction plot given a model and stock data.
        
        :param show_plot: whether plot should be displayed on-screen or just
            saved as a file
        :param save_plot: whether plot image should be saved to the disk
        :param filename: filename to give to saved plot (if plot is saved)
        """

        self._predict_scaled()
        # generate scaled forecast
        
        self._show_forecast(show_plot, save_plot, filename)
        # display final (unscaled) forecast plot