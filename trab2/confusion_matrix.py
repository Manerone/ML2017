from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt


if __name__ == '__main__':
    _, real_labels = load_svmlight_file('./teste.vet')

    file_path = './teste.vet.predict'
    title = 'LibSVM Confusion Matrix'
    save_path = 'confusion_matrixes/libsvm.png'

    # file_path = './scikit_svm_predicts'
    # title = 'Scikit Confusion Matrix'
    # save_path = 'confusion_matrixes/scikit.png'

    with open(file_path) as f:
        predicted_labels = f.readlines()

    predicted_labels = [float(x.strip()) for x in predicted_labels]

    skplt.plot_confusion_matrix(
        real_labels,
        predicted_labels,
        text_fontsize="large",
        normalize=True,
        title=title
    )
    plt.savefig(save_path, bbox_inches='tight')
