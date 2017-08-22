#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree



def main(X_data, y_data, test_size):

    # splits data
	X_train, X_test, y_train, y_test =  cross_validation.train_test_split(X_data, y_data, test_size=test_size)

	# cria uma DT
	clf  = tree.DecisionTreeClassifier()

	clf.fit(X_train, y_train)

	# predicao do classificador
	y_pred = clf.predict(X_test)

	# mostra o resultado do classificador na base de teste
	return clf.score(X_test, y_test)

if __name__ == "__main__":
	sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	X_data, y_data = load_svmlight_file('./data')
	for x in sizes:
		values = []
		for _ in xrange(0,10):
			values.append(main(X_data, y_data, x))
		print x, sum(values)/float(len(values))
