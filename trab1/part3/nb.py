#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt


def main(X_data, y_data, test_size):
    X_train, X_test, label_train, label_test = cross_validation.train_test_split(
        X_data, y_data, test_size=test_size
    )

    X_train = X_train.toarray()
    # cria o classificador

    gnb = BernoulliNB()

    gnb.fit(X_train, label_train)

    # predicao do classificador
    label_pred = gnb.predict(X_test)

    probs = gnb.predict_proba(X_test)

    correct_hist = []
    error_hist = []

    for i in xrange(0, len(label_pred)):
        max_prob = max(probs[i])
        if label_pred[i] == label_test[i]:
            correct_hist.append(max_prob)
        else:
            error_hist.append(max_prob)
    return correct_hist, error_hist


if __name__ == "__main__":
    X_data, y_data = load_svmlight_file('./data')
    correct, error = main(X_data, y_data, 0.5)

    n, bins, patches = plt.hist(correct, 50)

    plt.xlabel('Probability')
    plt.xlim(0, 1)
    plt.title('Histogram of probability over correct predictions')
    plt.grid(True)

    plt.savefig('nb_correct_hist.png', bbox_inches='tight')

    n, bins, patches = plt.hist(error, 50)

    plt.xlabel('Probability')
    plt.xlim(0, 1)
    plt.title('Histogram of probability over wrong predictions')
    plt.grid(True)

    plt.savefig('nb_error_hist.png', bbox_inches='tight')
