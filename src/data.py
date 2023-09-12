import pandas as pd
import numpy as np
from config import model_args
from typing import Dict

train_test_split = 0.7
windows_size = model_args.get("model_args").get('windows_size')
cut_num = 6


def create_blue_ball_data():
    pass


def create_train_test_data() -> Dict[str, Dict[str, str]]:
    source_data = pd.read_csv("./data/source_data.txt",
                              usecols=range(15), sep=" ",
                              header=None)
    train_data = source_data.iloc[:int(len(source_data) * train_test_split)]
    test_data = source_data.iloc[int(len(source_data) * train_test_split):]

    train_data = train_data.iloc[:, 2:].values
    test_data = test_data.iloc[:, 2:].values
    # logger.info("训练 集数据维度: {}".format(data.shape))

    train_x_data, train_y_data = [], []
    test_x_data, test_y_data = [], []

    for i in range(len(train_data) - windows_size - 1):
        sub_data = train_data[i:(i+windows_size+1), :]
        train_x_data.append(sub_data[1:])
        train_y_data.append(sub_data[0])

    for i in range(len(test_data) - windows_size - 1):
        sub_data = test_data[i:(i+windows_size+1), :]
        test_x_data.append(sub_data[1:])
        test_y_data.append(sub_data[0])

    return {
        "red_train_data": {
            "x_data": np.array(train_x_data)[:, :, :cut_num],
            "y_data": np.array(train_y_data)[:, :cut_num]
        },
        "red_test_data": {
            "x_data": np.array(test_x_data)[:, :, :cut_num],
            "y_data": np.array(test_y_data)[:, :cut_num]
        },
        "blue_train_data": {
            "x_data": np.array(train_x_data)[:, :, cut_num:cut_num+1],
            "y_data": np.array(train_y_data)[:, cut_num:cut_num+1]
        },
        "blue_test_data": {
            "x_data": np.array(test_x_data)[:, :, cut_num:cut_num+1],
            "y_data": np.array(test_y_data)[:, cut_num:cut_num+1]
        }
    }
