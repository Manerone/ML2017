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
	#print "Spliting data..."
	X_train, X_test, y_train, y_test =  cross_validation.train_test_split(X_data, y_data, test_size=test_size)

	# cria uma DT
	clf  = tree.DecisionTreeClassifier()

	clf.fit(X_train, y_train)

	# predicao do classificador
	y_pred = clf.predict(X_test)

	# cria a matriz de confusao
	return confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
	sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	X_data, y_data = load_svmlight_file('./data')
	print [{x: main(X_data, y_data, x)} for x in sizes]
