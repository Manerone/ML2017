#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB


import os


def main(X_data, y_data, test_size):
	print "Spliting data..."
	X_train, X_test, y_train, y_test =  cross_validation.train_test_split(X_data, y_data, test_size=test_size)

	X_train = X_train.toarray()
	# cria o classificador

	### Note that GaussianNB wont fit in this kind of sparse data (a lot of zero features)
	#gnb  = GaussianNB()
	gnb  = BernoulliNB()

	print 'Fitting NB'
	gnb.fit(X_train, y_train)

	# predicao do classificador
	print 'Predicting...'
	y_pred = gnb.predict(X_test)
	return confusion_matrix(y_test, y_pred)


if __name__ == "__main__":
	sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	print "Loading data..."
	X_data, y_data = load_svmlight_file('./data')
	print [{x: main(X_data, y_data, x)} for x in sizes]
