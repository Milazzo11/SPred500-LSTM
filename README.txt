A LSTM neural network time-series modeling application for forecasting
stock price data; implementing several learning and hyper-parameter 
optimization features, including a custom attention mechanism, early stopping, 
dropout regularization, and genetic algorithm hyper-parameter optimization.

Multiprocessing is also employed to increase program efficiency, and everything
is wrapped in an easy-to-use GUI.

To use:
=======
> Launch the SPred500 application
> Enter 1 or more ticker values (separated by spaces and/or commas)
> Enter a desired prediction period
> Check desired options
  - Genetic algorithm (GA) optimization will find optimal hyper-parameters for
    each specific problem and generate multiple viable models; this options
	is definitely better if accuracy and results are the goal, however, it can
	require a lot of computing power and time
> Click the "Generate Model"
> You plot(s) will appear in a labelled folder in the "plots" directory

RMSE:
=====
Root Mean Squared Error (RMSE) is the metric used to evaluate the trained models.
All reported RMSE values are normalized.

Files and directories:
======================
There are three subdirectories of "src":
> plots
  - Contains all generated plot forecasts
> models
  - Contains saved serialized GA models from preious run; these models can be
    recovered with some effort, but the models directory automatically clears
	on application startup
> resources
  - Contains application icon and preferences file (PREFS.txt) -- there are
    several advanced options that can be tweaked here (view documentation in
	PREFS file for more information)
	
Plot file naming:
=================
> DEFMODEL...
  - Model trained with default hyper-parameters
> GA_FINMODEL...
  - One of the models trained using GA optimized schema
> GA_TESTMODEL...
  - One of the models trained using schema from the last surviving GA generation
    if this option is enabled in the PREFS file
	
Warning flag:
=============
There is one warning flag that may display or show at the end of certain plot
files -- "N_lookback=1"; this indicates that genetic algorithm optimization
determined that the most optimal lookback period was just a single day.  This
is not necessarily a bad thing, but it is possible it could indicate volatility
and difficulty to find a pattern/make a prediction based on past data for a
particular stock.
	
Requirements:
=============
This application was built using Python 3.11 and all modules listed in the
requirements.txt file.  Although it is very likely it will still function with
no issues with a different version of Python or using different module versions
from those listed in the requirements file, this cannot be guaranteed.

Contact:
========
Please feel free to send me an email at maximusmilazzo@yahoo.com with any
questions or concerns.

Enjoy!