import pandas as pd


def get_train_data():
    return pd.read_csv("./input/train_data.csv")


# 下载历史的中奖数据并处理导出CSV到Input文件夹
def download_lottery_data():
    url = "http://e.17500.cn/getData/ssq.TXT"

    col = ["no", "date", "red_1", "red_2", "red_3", "red_4", "red_5", "red_6", "blue", "red_r_1", "red_r_2", "red_r_3",
           "red_r_4", "red_r_5", "red_r_6"]
    d = pd.read_table(url, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], sep=" ")
    d.columns = pd.Series(col)
    d.to_csv("./input/train_data.csv", index=False)


if __name__ == '__main__':
    download_lottery_data()
