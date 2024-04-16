import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import datetime

from utils import *
from classes import *

def tune_learning_rate(train_dataloader, val_dataloader, num_epochs):
    best_lr = 100000
    best_val_loss = 100000
    lr_list = [0.001, 0.01, 0.025, 0.050, 0.075, 0.1, 0.25, 0.5, 0.75, 0.99]

    # use cuda if available
    device = 'cuda'
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    for tune_lr in lr_list:
        model = Transformer(
          num_tokens=15000, dim_model=512, num_heads=8, num_encoder_layers=4, num_decoder_layers=4, dropout_p=0.1
        ).to(device)

        # Loss function -- critical for architecture analysis
        opt = torch.optim.SGD(model.parameters(), lr=tune_lr)
        loss_fn = nn.CrossEntropyLoss()

        train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs=num_epochs, device=device)

        plot_loss_curves(train_loss_list=train_loss_list, val_loss_list=validation_loss_list, num_epochs=num_epochs, filename="lr_results"+ str(tune_lr) +".png")

        if validation_loss_list[-1] < best_val_loss:
            print("***\n\n\nA final validation loss of: " + str(validation_loss_list[-1]) + " is better than the loss of: " + str(best_val_loss) + "\n\n\n***")
            best_lr = tune_lr
            best_val_loss = validation_loss_list[-1]

    print("***\n\n\nBest validation loss is: " + str(best_val_loss) + " Best lr is: " + str(best_lr) + "\n\n\n***")


def tune_attention_heads(train_dataloader, val_dataloader, num_epochs):
    best_attn_head = 100000
    best_val_loss = 100000
    attn_head_list = [1, 2, 4, 8, 16, 32]

    # use cuda if available
    device = 'cuda'
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    for tune_attn_head in attn_head_list:
        model = Transformer(
          num_tokens=15000, dim_model=512, num_heads=tune_attn_head, num_encoder_layers=4, num_decoder_layers=4, dropout_p=0.1
        ).to(device)

        # Loss function -- critical for architecture analysis
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss()

        train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs=num_epochs, device=device)

        plot_loss_curves(train_loss_list=train_loss_list, val_loss_list=validation_loss_list, num_epochs=num_epochs, filename="attn_head_results"+ str(tune_attn_head) +".png")

        if validation_loss_list[-1] < best_val_loss:
            print("***\n\n\nA final validation loss of: " + str(validation_loss_list[-1]) + " is better than the loss of: " + str(best_val_loss) + "\n\n\n***")
            best_attn_head = tune_attn_head
            best_val_loss = validation_loss_list[-1]

    print("***\n\n\nBest validation loss is: " + str(best_val_loss) + " Best attn head number is: " + str(best_attn_head) + "\n\n\n***")