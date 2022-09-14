import pandas as pd
import requests
from io import StringIO


def download_lottery_data():
    """
    Download the Lottery Data
    """

    url = "https://e.17500.cn/getData/ssq.TXT"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'
    }
    data = requests.get(url, headers=headers)

    col = [
        "no", "date",
        "red_1", "red_2", "red_3", "red_4", "red_5", "red_6",
        "blue",
        "red_r_1", "red_r_2", "red_r_3", "red_r_4", "red_r_5", "red_r_6"
    ]
    d = pd.read_csv(StringIO(data.text), usecols=[
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], sep=" ")
    d.columns = pd.Series(col)
    d.to_csv("./input/train_data.csv", index=False)


if __name__ == '__main__':
    download_lottery_data()
