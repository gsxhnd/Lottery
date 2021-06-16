import pandas as pd
import matplotlib.pyplot as plt


def get_train_data():
    train_data = pd.read_csv("./input/2020-06-02.csv")
    train_data.hist(bins=50, figsize=(20, 15))
    plt.show()
    return train_data


if __name__ == '__main__':
    get_train_data()
