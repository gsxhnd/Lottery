from keras.models import Model
from model_blue_ball import BlueBallModel
from config import model_args
import tensorflow as tf
import numpy as np


def train_blue_ball(train_x: np.ndarray, train_y: np.ndarray):
    blue_ball_model = BlueBallModel(
        batch_size=model_args["model_args"]["batch_size"],
        n_class=model_args["model_args"]["blue_n_class"],
        w_size=model_args["model_args"]["windows_size"],
        embedding_size=model_args["model_args"]["blue_embedding_size"],
        hidden_size=model_args["model_args"]["blue_hidden_size"],
        outputs_size=model_args["model_args"]["blue_n_class"],
        layer_size=model_args["model_args"]["blue_layer_size"]
    )
    model = Model(input=blue_ball_model.inputs)
    opt = tf.keras.optimizers.Adam(
        learning_rate=model_args["train_args"]["blue_learning_rate"],
        beta1=model_args["train_args"]["blue_beta1"],
        beta2=model_args["train_args"]["blue_beta2"],
        epsilon=model_args["train_args"]["blue_epsilon"],
        use_locking=False,
        name='Adam'
    ).minimize(blue_ball_model.loss)

    model.compile(optimizer=opt)
    model.fit(train_x, train_y)
