# A file to store helpful functions
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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

def get_multi_stock_dataframe(stocklist = ['MSFT', 'GOOG', 'AAPL'], filepath="7643_dataset/stocks"):
    dataframe_list = []
    for stock in stocklist:
        # Read each data frame in
        data_path = filepath + "/" + str(stock) + ".csv"
        print("Loading dataframe for: " + str(data_path))
        stock_df = pd.read_csv(data_path)
        # Appends to list for concatenation
        dataframe_list.append(stock_df)
        print("Appending stock_df:\n " + str(stock_df))
    # Concatenate
    print("Loaded all dataframes, concatenating...")
    result = pd.concat(dataframe_list, axis=0)
    print("Result of concat: \n" + str(result))
    return result

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
    if y_value != "":
        # generate y value as only one column
        print("Generating train test split... with split_num as: " + str(split_num))
        x_train = stock_dataframe["Date"].iloc[0:split_num]
        y_train = stock_dataframe[y_value].iloc[0:split_num]
        print(str(y_train))

        x_test = stock_dataframe["Date"].iloc[split_num:-1]
        y_test = stock_dataframe[y_value].iloc[split_num:-1]
        print(str(y_test))

        return x_train, y_train, x_test, y_test
    else:
        # generate x as dates, and ALL OTHER COLUMNS as y
        print("Generating train test split... with split_num as: " + str(split_num))
        x_train = stock_dataframe["Date"].iloc[0:split_num]
        y_train = stock_dataframe.iloc[0:split_num, 1:]
        print("y_train: \n" + str(y_train))

        x_test = stock_dataframe["Date"].iloc[split_num:-1]
        y_test = stock_dataframe.iloc[split_num:, 1:] #.iloc[0:split_num]
        print("y_test: \n" + str(y_test))

        return x_train, y_train, x_test, y_test

# Function to convert date column to seconds since the NYSE began
# Input df must have a 'Date' Column
def second_convert(input_df, reference_date = '1792-05-17'):
    # YYYY-MM-DD
    # NY Stock exchange started on May 17, 1792
    sec_df = (pd.to_datetime(input_df['Date']) - pd.to_datetime(reference_date)).dt.total_seconds()
    input_df['Date'] = sec_df
    return input_df

def day_convert(input_df, reference_date = '1985-12-31'):
    # YYYY-MM-DD
    # NY Stock exchange started on May 17, 1792
    input_df['Date'] = pd.to_datetime(input_df['Date'])
    day_df = (pd.to_datetime(input_df['Date']) - pd.to_datetime(reference_date)).dt.days
    input_df['Date'] = day_df
    return input_df

# Normalize if column too big for embeddings i.e. Volume data
# https://stackoverflow.com/questions/26414913/normalize-columns-of-a-dataframe
def normalize_column(input_df, col_to_normalize):
    print("Normalizing column: "+ col_to_normalize)
    normed = input_df[col_to_normalize] / input_df[col_to_normalize].max() * 10
    input_df[col_to_normalize] = normed

    return input_df


def plot_loss_curves(train_loss_list, val_loss_list, num_epochs, filename = ""):
    print("Plotting results and saving to file: " + str(filename))
    plt.figure()
    x_axis = np.arange(0, num_epochs, 1)
    y_val = val_loss_list
    y_train = train_loss_list
    plt.plot(x_axis, y_val, label = 'validation', marker='o')
    plt.plot(x_axis, y_train, label = 'train', marker='o')

    plt.title("Epochs vs loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(filename)




# inspired by: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
def train_loop(model, opt, loss_fn, dataloader, device):
    print("Executing training loop")

    model.train()
    total_loss = 0

    for batch in dataloader:
        X, y = batch[:, 0], batch[:, 1:]
        # print("training loop X: \n" + str(X))
        # print("training loop y: \n" + str(y))
        # If using cuda -- move to gpu
        #X, y = torch.tensor(X).to(torch.long).to(device), torch.tensor(y).to(torch.long).to(device)
        X, y = torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)
        X, y = X.to(device), y.to(device)

        # shift targets over by one to predict the token at pos = 1.
        # TODO verify empirically if we even need to do this for our application?
        encoded_inputs = model.encode(X)
        y_input = y[:, :-1]
        y_expected = y[:, 1:]
        encoded_y_inputs = model.encode(y_input)

        # Masking out next words
        # TODO enable this if necessary. May not be necesary for us.
        # seq_len = y_input.size(1)
        # tgt_mask = mode.get_tgt_mask(seq_len).to(device)
        
        # get prediction
        # print("Generating prediction... ")
        pred = model(encoded_inputs, encoded_y_inputs)    # source uses this -> pred = model(X, y_input, tgt_mask)
        # print("Generated prediction. Computing loss.")

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
            X, y = batch[:, 0], batch[:, 1:]
            #X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)
            X, y = torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)
            X, y = X.to(device), y.to(device)
            # Shift over one as we did in the training loop.
            # TODO Evaluate if this is necessary. Also add if else to cater if y is only one column
            encoded_inputs = model.encode(X)
            y_input = y[:, :-1]
            y_expected = y[:, 1:]
            encoded_y_inputs = model.encode(y_input)

            # mask out next words
            # TODO -- Evaluate if this is necessary
            # seq_len = y_input.size(1)
            # tgt_mask = model.get_tgt_mask(seq_len).to(device)

            # get prediction
            pred = model(encoded_inputs, encoded_y_inputs) # source example uses -> pred = model(X, y_input, tgt_mask)

            # permute to have batch size first
            pred = pred.permute(1, 2, 0)
            loss = loss_fn(pred, y_expected)

            total_loss += loss.detach().item()

    return total_loss / len(dataloader)

# Running training and validation -- see https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
# This function generates two lists, a training loss list and validation loss list. We can plot these.
def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, device):
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []

    print("Training and validating model")
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

        train_loss = train_loop(model, opt, loss_fn, train_dataloader, device=device)
        train_loss_list += [train_loss]

        validation_loss = validation_loop(model, loss_fn, val_dataloader, device=device)
        validation_loss_list += [validation_loss]

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()

    return train_loss_list, validation_loss_list

# https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
def batchify_data(data, batch_size=4, padding=False, padding_token=-1):
    print("Batching data...")
    print("len data: " + str(len(data)))
    batches = []
    for idx in range(0, len(data), batch_size):
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
            temp = np.array(data[idx : idx + batch_size])
            batches.append(temp) # batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))
    #print(str("Batch list: \n" + str(batches)))
    print(f"{len(batches)} batches of size {batch_size}")

    return batches


