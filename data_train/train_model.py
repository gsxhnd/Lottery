import pandas as pd

train_test_split = 0.7


# col_x = [
#     "no", "data",
#     "red_1", "red_2", "red_3",
#     "red_4", "red_5", "red_6",
#     "blue",
#     "red_r_1", "red_r_2", "red_r_3",
#     "red_r_4", "red_r_5", "red_r_6"
# ]

def create_train_test_data():
    source_data = pd.read_csv("./data/source_data.txt",
                              usecols=range(15), sep=" ",
                              header=None)
    train_data = source_data.iloc[:int(len(source_data) * train_test_split)]
    test_data = source_data.iloc[int(len(source_data) * train_test_split):]
    print(train_data)
    print(test_data)


def train_model(train_x, train_y, predict_x):
    pass
    # liner = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
    # liner.fit(train_x, train_y)
    # predict = liner.predict(predict_x)
    # return predict


if __name__ == '__main__':
    create_train_test_data()
