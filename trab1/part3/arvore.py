#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import os
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
from sklearn import tree


def main(X_data, y_data, test_size):

    X_train, X_test, label_train, label_test = cross_validation.train_test_split(
        X_data, y_data, test_size=0.5
    )

    # cria uma DT
    clf = tree.DecisionTreeClassifier()

    clf.fit(X_train, label_train)

    label_pred = clf.predict(X_test)

    ### Probabilities
    probs = clf.predict_proba(X_test)

    correct_hist = []
    error_hist = []

    for i in xrange(0, len(label_pred)):
        max_prob = max(probs[i])
        if label_pred[i] == label_test[i]:
            correct_hist.append(max_prob)
        else:
            error_hist.append(max_prob)


if __name__ == "__main__":
    X_data, y_data = load_svmlight_file(os.getcwd() + '/./data')
    main(X_data, y_data, 0.5)
