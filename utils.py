# A file to store helpful functions
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

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

# inspired by: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
def train_loop(model, opt, loss_fn, dataloader, device):
    print("Executing training loop")

    model.train()
    total_loss = 0

    for batch in dataloader:
        X, y = batch[:, 0], batch[:, 1]
        # If using cuda -- move to gpu
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

        # shift targets over by one to predict the token at pos = 1.
        # TODO verify empirically if we even need to do this for our application?

        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        # Masking out next words
        # TODO enable this if necessary. May not be necesary for us.
        # seq_len = y_input.size(1)
        # tgt_mask = mode.get_tgt_mask(seq_len).to(device)

        # get prediction
        pred = model(X, y_input)    # source uses this -> pred = model(X, y_input, tgt_mask)

        # change pred to have batch size first
        pred = pred.permute(1, 2, 0)

        # Compute the loss
        loss = loss_fn(pred, y_expected)

        # Backpropogation
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)

# source https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
def validation_loop(model, loss_fn, dataloader, device):
    model.eval() # set model to evaluation mode
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

            # Shift over one as we did in the training loop.
            # TODO Evaluate if this is necessary.

            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            # mask out next words
            # TODO -- Evaluate if this is necessary
            seq_len = y_input.size(1)
            tgt_mask = model.get_tgt_mask(seq_len).to(device)

            # get prediction
            pred = model(X, y_input) # source example uses -> pred = model(X, y_input, tgt_mask)

            # permute to have batch size first
            pred = pred.permute(1, 2, 0)
            loss = loss_fn(pred, y_expected)

            total_loss += loss.detach().item()

    return total_loss / len(dataloader)

# Running training and validation -- see https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
# This function generates two lists, a training loss list and validation loss list. We can plot these.
def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []

    print("Training and validating model")
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]

        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()

    return train_loss_list, validation_loss_list

# https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
def batchify_data(data, batch_size=16, padding=False, padding_token=-1):
    print("Batching data...")
    batches = []
    for idx in range(0, len(data), batch_size):
        print("idx: " + str(idx))
        print("len data: " + str(len(data)))
        # We make sure we dont get the last bit if its not batch_size size
        if idx + batch_size < len(data):
            # Here you would need to get the max length of the batch,
            # and normalize the length with the PAD token.
            if padding:
                max_batch_length = 0

                # Get longest sentence in batch
                for seq in data[idx : idx + batch_size]:
                    if len(seq) > max_batch_length:
                        max_batch_length = len(seq)

                # Append X padding tokens until it reaches the max length
                for seq_idx in range(batch_size):
                    remaining_length = max_batch_length - len(data[idx + seq_idx])
                    data[idx + seq_idx] += [padding_token] * remaining_length

            batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))

    print(f"{len(batches)} batches of size {batch_size}")

    return batches


