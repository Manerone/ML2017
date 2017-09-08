from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from numpy import argmax


if __name__ == '__main__':
    c = 8.0
    gamma = 2.0
    train_attrs, train_labels = load_svmlight_file('./treino.vet')
    test_attrs, test_labels = load_svmlight_file('./teste.vet')

    classificator = SVC(C=c, gamma=gamma, probability=True)
    classificator.fit(train_attrs, train_labels)
    probs = classificator.predict_proba(test_attrs)

    for prob in probs:
        label = argmax(prob)
        string = ' '.join([str(x) for x in prob])
        print(str(label) + ' ' + string)
