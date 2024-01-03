from keras.models import Model
import tensorflow as tf
import numpy as np

# from model_blue_ball import BlueBallModel
from config import model_args
from data import create_train_test_data

inputs = tf.keras.layers.Input(
    shape=(model_args["model_args"]["windows_size"], ),
    batch_size=model_args["model_args"]["batch_size"],
    dtype=tf.int32, name="inputs"
)
tag_indices = tf.keras.layers.Input(
    shape=(model_args["model_args"]["blue_n_class"], ),
    batch_size=model_args["model_args"]["batch_size"],
    dtype=tf.float32,
    name="tag_indices"
)
embedding = tf.keras.layers.Embedding(
    model_args["model_args"]["blue_n_class"],
    model_args["model_args"]["blue_embedding_size"])(inputs)

lstm = tf.keras.layers.LSTM(
    model_args["model_args"]["blue_hidden_size"],
    return_sequences=True)(embedding)

for _ in range(model_args["model_args"]["blue_layer_size"]):
    lstm = tf.keras.layers.LSTM(
        model_args["model_args"]["blue_hidden_size"],
        return_sequences=True)(lstm)

final_lstm = tf.keras.layers.LSTM(
    model_args["model_args"]["blue_hidden_size"], recurrent_dropout=0.2)(lstm)

outputs = tf.keras.layers.Dense(
    model_args["model_args"]["blue_n_class"], activation="softmax")(final_lstm)

loss = tf.reduce_sum(tag_indices * tf.math.log(outputs))


def train_blue_ball(train_x: np.ndarray, train_y: np.ndarray):
    model = Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(
        # learning_rate=model_args["train_args"]["blue_learning_rate"],
        # beta1=model_args["train_args"]["blue_beta1"],
        # beta2=model_args["train_args"]["blue_beta2"],
        # epsilon=model_args["train_args"]["blue_epsilon"],
        # use_locking=False,
        name='Adam'
    ).minimize(loss=loss, var_list="")

    model.compile(optimizer=opt)
    model.fit(train_x, train_y)


if __name__ == "__main__":
    data = create_train_test_data()
    train_blue_ball(data["blue_train_data"]["x_data"],
                    data["blue_train_data"]["y_data"])
