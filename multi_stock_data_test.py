# Runner to verify funtionality of the multi stock training
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import datetime

from utils import *
from classes import *
from tuning import *

if __name__=="__main__":
    # test_configuration()
    # Todo establish training and validation datasets -- remember to set seed to ensure batching is repeatable

    stock_list = ['MSFT', 'AMZN', 'GOOG']
    combined_df = get_multi_stock_dataframe(stocklist=stock_list, filepath="7643_dataset/stocks")

    # Convert dates to seconds or days
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print("Raw dataframe: \n" + str(combined_df))
    print("Converting dates to seconds...")
    # combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    day_data = day_convert(combined_df, reference_date='1985-12-31')
    # print("Sec data: " + str(sec_data))

    combined_df = day_data
    normed_volume = normalize_column(combined_df, 'Volume')
    combined_df = normed_volume
    print("Date data converted to days. New df: \n" + str(combined_df))

    # train test split
    comb_x_train, comb_y_train, comb_x_test, comb_y_test = simple_time_split_validation(stock_dataframe=combined_df, split_num=.75, y_value="")

    # Concatenate data in the form [X, y]

    training_data = [comb_x_train, comb_y_train]
    training_data = pd.concat(training_data, axis=1)
    validation_data = [comb_x_test, comb_y_test]
    validation_data = pd.concat(validation_data, axis=1)

    # use cuda if available
    device = 'cuda'
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = Transformer(
        num_tokens=15000, dim_model=512, num_heads=1, num_encoder_layers=1, num_decoder_layers=1, dropout_p=0.1
    ).to(device)

    # Loss function -- critical for architecture analysis
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    print("Model generated.")

    train_dataloader = batchify_data(training_data, batch_size=4)
    val_dataloader = batchify_data(validation_data, batch_size=4)
    print("Data has been loaded.")

    print("Computing the training and validation losses.")
    num_epochs = 5
    train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader,
                                                epochs=num_epochs, device=device)
    plot_loss_curves(train_loss_list=train_loss_list, val_loss_list=validation_loss_list, num_epochs=num_epochs,
                     filename='results.png')
