from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt


if __name__ == '__main__':
    _, real_labels = load_svmlight_file('./teste.vet')

    with open('./teste.vet.predict') as f:
        predicted_labels = f.readlines()

    # with open('./scikit_svm_predicts') as f:
    #     predicted_labels = f.readlines()

    predicted_labels = [float(x.strip()) for x in predicted_labels]

    skplt.plot_confusion_matrix(
        real_labels,
        predicted_labels,
        text_fontsize="large",
        normalize=True,
        title='LibSVM Confusion Matrix'
    )
    plt.savefig('./confusion-matrix-libsvm.png', bbox_inches='tight')
