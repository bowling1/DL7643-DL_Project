# A file to store helpful functions
import pandas as pd
import numpy as np

# KAA 04_08_2024 -- Feel free to edit / Augment as needed!
# This function takes a stock or etf as input. Be sure to set the filepath correctly if using a stock or etf.
# This assumes this dataset as stated in the proposal: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset?resource=download
def get_dataframe(stock_name = "SPY", filepath = "7643_dataset/etfs"):
    # function to read in data as a dataframe using Pandas
    try:
        data_path = filepath + "/" + stock_name + ".csv"
        print("Loading data for: " + stock_name + " at " + data_path)
        stock_dataframe = pd.read_csv(data_path)
        print("Successfully loaded data.")
        return stock_dataframe
    except:
        print("Failed to load data. Verify stock name exists in datapath.")

# Use this to verify the configuration. This simple script just makes sure that you can load and print data.
def test_configuration():
    try:
        # Load data into dataframes from csv files
        print("Loading data")
        SPY_data = get_dataframe(stock_name="SPY", filepath="7643_dataset/etfs")
        MSFT_data = get_dataframe(stock_name="MSFT", filepath="7643_dataset/stocks")

        # Verify data is loaded and can be printed
        print("SPY DATA: \n" + str(SPY_data))
        print("MSFT DATA: \n" + str(MSFT_data))

        # get statistics of various columns (to ensure that column names are working properly)
        print("Mean of Open column for MSFT is: " + str(MSFT_data["Open"].mean()))
        print("Max of Volume column for MSFT is: " + str(MSFT_data["Open"].max()))
        print("Statistics computed from " + str(MSFT_data["Date"].iloc[0]) + " To " + str(MSFT_data["Date"].iloc[-1]))
        print("Configuration passed.")
    except:
        # Message to indicate something may be wrong with file structure
        print("Configuration failed.")

# Strategies for splitting into train and validation sets
# Refer to https://forecastegy.com/posts/time-series-cross-validation-python/#:~:text=You%20pick%20a%20time%20point,of%20your%20data%20as%20training.

# simple_time_split_validation - this method picks a date in the range and sets the range before that as the train set and after
# as the validation set. Takes argument split_num to indicate what iloc to be the cutoff for the train-test set.

def simple_time_split_validation(stock_dataframe, seed = 0, split_num = 0.75, y_value = ""):
    np.random.seed(seed)  # set seed if necessary, default to 0
    if split_num < 1: # If split_num is < 1, interprets the value as a percentage of the data to set as training set.
        percent = split_num * 100
        print("Using " + str(percent) + " Percent of the dataset for training and " + str(100-percent) + " for testing.")
        split_num = int(split_num*stock_dataframe.shape[0])
    print("Generating train test split... with split_num as: " + str(split_num))
    x_train = stock_dataframe["Date"].iloc[0:split_num]
    y_train = stock_dataframe[y_value].iloc[0:split_num]
    print(str(y_train))

    x_test = stock_dataframe["Date"].iloc[split_num:-1]
    y_test = stock_dataframe[y_value].iloc[split_num:-1]
    print(str(y_test))

    return x_train, y_train, x_test, y_test