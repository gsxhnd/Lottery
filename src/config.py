model_args: dict[str, dict] = {
    "raw_data": "./data/raw_data.txt",
    "model_args": {
        "windows_size": 3,
        "batch_size": 1,
        "sequence_len": 6,
        "red_n_class": 33,
        "red_epochs": 1,
        "red_embedding_size": 32,
        "red_hidden_size": 32,
        "red_layer_size": 1,
        "blue_n_class": 16,
        "blue_epochs": 128,
        "blue_embedding_size": 32,
        "blue_hidden_size": 32,
        "blue_layer_size": 1
    },
    "train_args": {
        "red_learning_rate": 0.001,
        "red_beta1": 0.9,
        "red_beta2": 0.999,
        "red_epsilon": 1e-08,
        "blue_learning_rate": 0.001,
        "blue_beta1": 0.9,
        "blue_beta2": 0.999,
        "blue_epsilon": 1e-08
    },
    "path": {
        "red": "./output/red_ball_model/",
        "blue": "./output/blue_ball_model/",
    }
}
