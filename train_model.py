from get_train_data import *
from sklearn.linear_model import LinearRegression


def train_model(train_x, train_y, predict_x):
    liner = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
    liner.fit(train_x, train_y)
    predict = liner.predict(predict_x)
    return predict


if __name__ == '__main__':
    train_data = get_train_data()
    col_x = ["red_1", "red_2", "red_3", "red_4", "red_5", "red_6", "blue", "red_r_1", "red_r_2", "red_r_3", "red_r_4",
             "red_r_5", "red_r_6"]
    x = train_data.drop(columns=col_x).copy()
    x["date"] = pd.to_datetime(x["date"])

    col_y = ["no", "date", "red_r_1", "red_r_2", "red_r_3", "red_r_4", "red_r_5", "red_r_6"]
    y = train_data.drop(columns=col_y, axis=1).copy()
    for i in y:
        print("ball number: ")
        print(y[i])

    df = pd.DataFrame({"no": ["2021095"]})
    df["date"] = pd.to_datetime(["2021-08-22"])

    print(x)
    print(y)
    print(df)

    p = train_model(x, y, df)
    print(p)
