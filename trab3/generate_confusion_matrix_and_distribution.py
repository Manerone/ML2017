import matplotlib.pyplot as plt
import scikitplot.plotters as skplt


def read_file(path):
    real = []
    predicted = []
    distributions = []
    with open(path) as f:
        for line in f:
            r, p, d = line.strip().split(' ')
            real.append(r)
            predicted.append(p)
            distributions.append(d)
    return real, predicted, distributions


def generate_confusion_matrix(real_labels, predicted_labels):
    skplt.plot_confusion_matrix(
        real_labels, predicted_labels,
        normalize=True,
        title='Normalized Confusion Matrix',
        text_fontsize="large"
    )
    plt.savefig('confusion_matrix.png', bbox_inches='tight')


def generate_distribution(real_labels, predicted_labels, distributions):
    correct = []
    error = []

    values = zip(real_labels, predicted_labels, distributions)
    for real, predicion, distribution in values:
        if real == predicion:
            correct.append(float(distribution))
        else:
            error.append(float(distribution))

    plt.figure()
    n, bins, patches = plt.hist(
        [correct, error],
        bins=10,
        label=['Acertos', 'Erros'],
        stacked=False, fill=True
    )

    plt.xlabel('Probabilidades')
    plt.xlim(0, 1.1)
    plt.title('Distribuicao de probabilidades')
    plt.grid(True)
    plt.legend(prop={'size': 10}, loc='upper center')

    plt.savefig('distribution.png', bbox_inches='tight')


if __name__ == '__main__':
    real_labels, predicted_labels, distributions = read_file('./output.txt')

    generate_confusion_matrix(real_labels, predicted_labels)
    generate_distribution(real_labels, predicted_labels, distributions)
