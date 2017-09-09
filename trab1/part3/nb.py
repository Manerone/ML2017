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


def separete_in_bins(array):
    splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # bins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    hist = []
    for value in array:
        for i in xrange(0, len(splits)):
            if value <= splits[i]:
                hist.append(splits[i])
    return hist


if __name__ == "__main__":
    X_data, y_data = load_svmlight_file('./data')
    correct, error = main(X_data, y_data, 0.5)

    n, bins, patches = plt.hist(
        [correct, error],
        bins=10,
        label=['Acertos', 'Erros'],
        stacked=False, fill=True
    )

    plt.xlabel('Probabilidades')
    plt.title('Histograma de probabilidades (Naive Bayes)')
    plt.grid(True)
    plt.legend(prop={'size': 10}, loc='upper center')

    plt.savefig('nb_hist.png', bbox_inches='tight')
