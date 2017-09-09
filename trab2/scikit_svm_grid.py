from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_svmlight_file


def svc_param_selection(train_features, train_labels):
    Cs = [2**x for x in xrange(-5, 17, 2)]
    gammas = [2**x for x in xrange(-15, 5, 2)]
    param_grid = {'C': Cs, 'gamma': gammas}
    gs = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, n_jobs=4)
    gs.fit(train_features, train_labels)
    gs.best_params_
    return gs.best_params_


if __name__ == '__main__':
    train_features, train_labels = load_svmlight_file('./treino.vet')
    print svc_param_selection(train_features, train_labels)
