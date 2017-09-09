# def construct_dict(array):
#     positions = zip(array, range(0, len(array)))
#     return {key: value for (key, value) in positions}
#
#
# if __name__ == '__main__':
#     file_path = './libsvm_prob_output'
#     with open(file_path) as f:
#         label_location = construct_dict(f.readline().split(' '))
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt


if __name__ == '__main__':
    _, real_labels = load_svmlight_file('./teste.vet')

    file_path = './libsvm_prob_output'
    title = 'LibSVM Confusion Matrix Using Probability'
    save_path = 'confusion_matrixes/libsvm_prob.png'

    # file_path = './scikit_svm_prob_output'
    # title = 'Scikit Confusion Matrix Using Probability'
    # save_path = 'confusion_matrixes/scikit_prob.png'

    with open(file_path) as f:
        lines = f.readlines()

    lines.pop(0)  # delete labels location
    predicted_labels = [float(x[0]) for x in lines]

    skplt.plot_confusion_matrix(
        real_labels,
        predicted_labels,
        text_fontsize="large",
        normalize=True,
        title=title
    )
    plt.savefig(save_path, bbox_inches='tight')
