import pandas as pd
import numpy as np
import tensorflow as tf
from config import model_args
from model import SignalLstmModel
from typing import Dict

train_test_split = 0.7
windows_size = model_args.get("model_args").get('windows_size')
cut_num = 6


# col_x = [
#     "no", "data",
#     "red_1", "red_2", "red_3",
#     "red_4", "red_5", "red_6",
#     "blue",
#     "red_r_1", "red_r_2", "red_r_3",
#     "red_r_4", "red_r_5", "red_r_6"
# ]

def create_train_test_data() -> Dict[str, Dict[str, str]]:
    source_data = pd.read_csv("./data/source_data.txt",
                              usecols=range(15), sep=" ",
                              header=None)
    train_data = source_data.iloc[:int(len(source_data) * train_test_split)]
    test_data = source_data.iloc[int(len(source_data) * train_test_split):]

    train_data = train_data.iloc[:, 2:].values
    test_data = test_data.iloc[:, 2:].values
    # logger.info("训练集数据维度: {}".format(data.shape))

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
            "x_data": np.array(train_x_data)[:, :, :cut_num],
            "y_data": np.array(train_y_data)[:, :cut_num]
        },
        "blue_test_data": {
            "x_data": np.array(test_x_data)[:, :, :cut_num],
            "y_data": np.array(test_y_data)[:, :cut_num]
        }
    }


def train_blue_ball_mode(train_x, train_y, test_x, test_y):
    train_x = train_x - 1
    train_data_len = train_x.shape[0]
    # train_x = train_x - 1
    # train_y = train_y - 1
    # train_data_len = train_x.shape[0]

    # test_x = test_x - 1
    # test_y = test_y - 1
    # test_data_len = test_x.shape[0]

    with tf.compat.v1.Session() as session:
        blue_ball_model = SignalLstmModel(
            batch_size=model_args["model_args"]["batch_size"],
            n_class=model_args["model_args"]["blue_n_class"],
            w_size=model_args["model_args"]["windows_size"],
            embedding_size=model_args["model_args"]["blue_embedding_size"],
            hidden_size=model_args["model_args"]["blue_hidden_size"],
            outputs_size=model_args["model_args"]["blue_n_class"],
            layer_size=model_args["model_args"]["blue_layer_size"]
        )
        train_step = tf.compat.v1.train.AdamOptimizer(
            learning_rate=model_args["train_args"]["blue_learning_rate"],
            beta1=model_args["train_args"]["blue_beta1"],
            beta2=model_args["train_args"]["blue_beta2"],
            epsilon=model_args["train_args"]["blue_epsilon"],
            use_locking=False,
            name='Adam'
        ).minimize(blue_ball_model.loss)

        session.run(tf.compat.v1.global_variables_initializer())
        # sequence_len = ""
        for epoch in range(model_args["model_args"]["blue_epochs"]):
            for i in range(train_data_len):
                _, loss_, pred = session.run(
                    [
                        train_step,
                        blue_ball_model.loss,
                        blue_ball_model.pred_label
                    ],
                    feed_dict={
                        "inputs:0": train_x[i:(i+1), :],
                        "tag_indices:0": train_y[i:(i+1), :],
                    }
                )
                if i % 100 == 0:
                    print("epoch: {}, loss: {}".format(epoch, loss_))


if __name__ == '__main__':
    train_test_data = create_train_test_data()

    train_blue_ball_mode(
        train_test_data["blue_train_data"]["x_data"],
        train_test_data["blue_train_data"]["y_data"],
        train_test_data["blue_test_data"]["x_data"],
        train_test_data["blue_test_data"]["y_data"]
    )
