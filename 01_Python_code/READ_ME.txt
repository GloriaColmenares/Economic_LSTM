1- install all the packages in requirements.txt
2- in paths.py change the BASE_INPUT_PATH and BASE_OUTPUT_PATH according to your directory
3- open each of the models' files in 'deep_learning_methods' folder and click 'run all' to get the results of each models (all 3 models together may take around 230 minutes)
4- do the same for the files in 'statistical_methods' folder (all 8 models together may take around 242 minutes)
5- after producing all the results, open 'results_postprocessing.ipynb' in 'post_processing' folder to calculate the errors and other metrics
6- open 'results_visualisation.ipynb' in 'post_processing' folder to produce the plot which exist in the last meeting's ppt file

! all the steps should be done in order, or else you may face an error.
! if you want to have the newest market prices, do as follows as a step before mentioned step 3:
    - from 'data' folder, open 'data_creation.ipynb' and click 'run all' to download the data from website and make the dataframe
! exogenous factors data are only provided till the end of Feb 2025, so if you run with data later than that month, models with exogenous factors (ARMAX, SARMAX, LSTM with feature) will face an error as they have missing data.
! the running time depends on the system that the models get run on it. the times given are with a cluster that has 48 processors 3.85GHz and 256GB RAM and 48GB GPU memory.
! the functions in data_processing.py, data_reading.py, and helper_functions.py has been used throughout other files. so they will be used inderectly and no need to run them seperately.