# Runner to verify funtionality of the Utils.py file
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

    MSFT_data = get_dataframe(stock_name="MSFT", filepath="7643_dataset/stocks")

    # Convert dates to seconds or days
    print("Raw dataframe: \n" + str(MSFT_data))
    print("Converting dates to seconds...")
    MSFT_data['Date'] = pd.to_datetime(MSFT_data['Date'])

    day_data = day_convert(MSFT_data)
    # print("Sec data: " + str(sec_data))
    MSFT_data = day_data
    normed_volume = normalize_column(MSFT_data, 'Volume')
    MSFT_data = normed_volume
    print("Date data converted to days. New df: \n" + str(MSFT_data))

    # train test split
    MSFT_x_train, MSFT_y_train, MSFT_x_test, MSFT_y_test = simple_time_split_validation(stock_dataframe=MSFT_data, split_num=.75, y_value="")

    # Concatenate data in the form [X, y]

    training_data = [MSFT_x_train, MSFT_y_train]
    training_data = pd.concat(training_data, axis=1)
    validation_data = [MSFT_x_test, MSFT_y_test]
    validation_data = pd.concat(validation_data, axis=1)

    print("Training data:\n " + str(training_data))
    print("Validation data:\n " + str(validation_data))
    print("Testing Transformer...")

    # use cuda if available
    device = 'cuda'
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = Transformer(
          num_tokens=15000, dim_model=512, num_heads=8, num_encoder_layers=4, num_decoder_layers=4, dropout_p=0.1
    ).to(device)

    # Loss function -- critical for architecture analysis
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    print("Model generated.")

    train_dataloader = batchify_data(training_data, batch_size=128)
    val_dataloader = batchify_data(validation_data, batch_size=128)
    print("Data has been loaded.")

    print("Computing the training and validation losses.")
    num_epochs = 20
    train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs=num_epochs, device=device)
    plot_loss_curves(train_loss_list=train_loss_list, val_loss_list=validation_loss_list, num_epochs=num_epochs, filename='results.png')
    

    # Tuning functions
    
    print("Tuning learning rate...")
    tune_learning_rate(train_dataloader=train_dataloader, val_dataloader=val_dataloader, num_epochs=num_epochs)
    print("Tuning attention heads...")
    tune_attention_heads(train_dataloader=train_dataloader, val_dataloader=val_dataloader, num_epochs=num_epochs)

    print("Train and testing complete.")