from get_train_data import *

if __name__ == '__main__':
    train_data = get_train_data()
    train_data.hist(bins=50, figsize=(20, 15))
    plt.show()
