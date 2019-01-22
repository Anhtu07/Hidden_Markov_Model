from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6)
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Ma trận nhầm lẫn',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt if int(cm[i, j]) != cm[i, j] else '.0f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Nhãn thật')
    plt.xlabel('Nhãn đoán')


def dz(data_loader, model):
    true_labels = data_loader.test_labels
    predicted_labels = []
    test_features = data_loader.test_features
    labels = ['khong', 'mot', 'hai', 'ba', 'bon', 'nam', 'sau', 'bay', 'tam', 'chin']
    for idx in range(len(test_features)):
        predicted_labels.append(model.get_class(data_loader.test_features[idx]))
    print(true_labels)
    print(predicted_labels)
    con = confusion_matrix(true_labels, predicted_labels, labels=labels)
    plot_confusion_matrix(con, labels)
