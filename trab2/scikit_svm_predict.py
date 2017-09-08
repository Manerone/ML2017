from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file


if __name__ == '__main__':
    c = 8.0
    gamma = 2
    train_attrs, train_labels = load_svmlight_file('./treino.vet')
    test_attrs, test_labels = load_svmlight_file('./teste.vet')

    classificator = SVC(C=c, gamma=gamma)
    classificator.fit(train_attrs, train_labels)
    print 'Prediction score:', classificator.score(test_attrs, test_labels)
    predicts = classificator.predict(test_attrs)
    for predict in predicts:
        print predict
