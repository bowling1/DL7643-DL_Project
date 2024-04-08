# A file to store helpful functions
import pandas as pd

# KAA 04_08_2024 -- Feel free to edit / Augment as needed!
# This function takes a stock or etf as input. Be sure to set the filepath correctly if using a stock or etf.
# This assumes this dataset as stated in the proposal: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset?resource=download
def get_dataframe(stock_name = "SPY", filepath = "7643_dataset/etfs"):
    # function to read in data as a dataframe using Pandas
    data_path = filepath + "/" + stock_name + ".csv"
    print("Loading data for: " + stock_name + " at " + data_path)
    stock_dataframe = pd.read_csv(data_path)

    return stock_dataframe
