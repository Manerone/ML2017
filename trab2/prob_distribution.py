from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt


def construct_dict(array):
    positions = zip(array, range(0, len(array)))
    return {float(key): value for (key, value) in positions}


def build_hist(file_path, save_path, real_labels):
    with open(file_path) as f:
        label_location = construct_dict(f.readline().strip().split(' '))
        lines = f.readlines()

    correct = []
    error = []

    for real_label, line in zip(real_labels, lines):
        line = line.strip().split(' ')
        predicted = float(line.pop(0))
        prob = float(line[label_location[predicted]])
        if real_label == predicted:
            correct.append(prob)
        else:
            error.append(prob)

    plt.figure()
    n, bins, patches = plt.hist(
        [correct, error],
        bins=10,
        label=['Acertos', 'Erros'],
        stacked=False, fill=True
    )

    plt.xlabel('Probabilidades')
    plt.xlim(0, 1.1)
    plt.title('Histograma de probabilidades')
    plt.grid(True)
    plt.legend(prop={'size': 10}, loc='upper center')

    plt.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
    _, real_labels = load_svmlight_file('./teste.vet')

    file_path = './libsvm_prob_output'
    save_path = 'prob_dist/libsvm.png'

    build_hist(file_path, save_path, real_labels)

    file_path = './scikit_svm_prob_output'
    save_path = 'prob_dist/scikit.png'

    build_hist(file_path, save_path, real_labels)
