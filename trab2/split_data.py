from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file


if __name__ == '__main__':
    X_data, y_data = load_svmlight_file('./data')

    train_features = X_data[:2000]
    train_labels = y_data[:2000]

    test_features = X_data[2000:]
    test_labels = y_data[2000:]

    dump_svmlight_file(train_features, train_labels, './train_base')

    dump_svmlight_file(test_features, test_labels, './test_base')
