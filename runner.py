# Runner to verify funtionality of the Utils.py file
from utils import *

if __name__=="__main__":
    print("Getting data")
    SPY_data = get_dataframe(stock_name="SPY", filepath="7643_dataset/etfs")
    MSFT_data = get_dataframe(stock_name="MSFT", filepath="7643_dataset/stocks")
    print("SPY DATA: \n" + str(SPY_data))
    print("MSFT DATA: \n" + str(MSFT_data))