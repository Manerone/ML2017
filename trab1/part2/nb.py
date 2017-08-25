#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import os


def main(X_data, y_data, test_size):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X_data, y_data, test_size=test_size
    )

    X_train = X_train.toarray()

    # cria o classificador
    gnb = BernoulliNB()

    gnb.fit(X_train, y_train)

    # predicao do classificador
    y_pred = gnb.predict(X_test)

    return y_test, y_pred


if __name__ == "__main__":
    sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    X_data, y_data = load_svmlight_file('./data')
    if not os.path.exists('./nb/'):
        os.makedirs('./nb/')
    for x in sizes:
        y_test, y_pred = main(X_data, y_data, x)
        skplt.plot_confusion_matrix(y_test, y_pred, normalize=True)
        plt.savefig('./nb/' + str(int(x*10)) + '.png', bbox_inches='tight')
