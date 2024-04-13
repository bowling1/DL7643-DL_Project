# Runner to verify funtionality of the Utils.py file
from utils import *
from classes import *

if __name__=="__main__":
    test_configuration()
    # Todo establish training and validation datasets -- remember to set seed to ensure batching is repeatable

    MSFT_data = get_dataframe(stock_name="MSFT", filepath="7643_dataset/stocks")
    MSFT_x_train, MSFT_y_train, MSFT_x_test, MSFT_y_test = simple_time_split_validation(stock_dataframe=MSFT_data, split_num=.75, y_value="Volume")

    print("")

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = Transformer(
    #     num_tokens=4, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1
    # ).to('cpu')
    #
    # # Loss function -- critical for architecture analysis
    # opt = torch.optim.SGD(model.parameters(), lr=0.01)
    # loss_fn = nn.CrossEntropyLoss()

