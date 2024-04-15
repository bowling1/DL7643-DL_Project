# Runner to verify funtionality of the Utils.py file
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import datetime

from utils import *
from classes import *



if __name__=="__main__":
    # test_configuration()
    # Todo establish training and validation datasets -- remember to set seed to ensure batching is repeatable

    MSFT_data = get_dataframe(stock_name="MSFT", filepath="7643_dataset/stocks")

    # Convert dates to seconds
    print("Raw dataframe: \n" + str(MSFT_data))
    print("Converting dates to seconds...")
    MSFT_data['Date'] = pd.to_datetime(MSFT_data['Date'])

    day_data = day_convert(MSFT_data)
    # print("Sec data: " + str(sec_data))
    MSFT_data = day_data
    print("Date data converted to seconds. New df: \n" + str(MSFT_data))

    MSFT_x_train, MSFT_y_train, MSFT_x_test, MSFT_y_test = simple_time_split_validation(stock_dataframe=MSFT_data, split_num=.75, y_value="")

    # Concatenate data in the form [X, y]

    training_data = [MSFT_x_train, MSFT_y_train]
    training_data = pd.concat(training_data, axis=1)
    validation_data = [MSFT_x_test, MSFT_y_test]
    validation_data = pd.concat(validation_data, axis=1)
    #
    print("Training data:\n " + str(training_data))
    print("Validation data:\n " + str(validation_data))
    print("Testing Transformer...")
    #
    #
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(
          num_tokens=100000, dim_model=1000, num_heads=1, num_encoder_layers=1, num_decoder_layers=1, dropout_p=0.1
    ).to('cpu')
    #
    print("Model generated.")
    #
    # Loss function -- critical for architecture analysis
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    print("Generating the training and validation losses.")
    train_dataloader = batchify_data(training_data)
    val_dataloader = batchify_data(validation_data)
    print("Data has been loaded.")

    train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, 10, device='cpu')

    print("Train and testing complete.")