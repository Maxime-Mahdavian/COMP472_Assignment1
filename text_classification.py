import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import sklearn as sk
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.model_selection as model_selection
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as metric
from collections import Counter


def get_distribution():
    dist = {}

    for folder in os.listdir('BBC'):
        if os.path.isfile(os.path.join('BBC', folder)):
            continue
        else:
            dist[folder] = len([name for name in os.listdir('BBC/' + folder)])

    return dist


def plot_distribution():
    dist = get_distribution()
    # print(dist)

    plt.bar(list(dist.keys()), dist.values())
    plt.title("File distribution in for every category")
    # plt.show()
    plt.savefig("BBC-distribution.pdf")


def prepare_datasets():
    files = load_files('BBC', encoding="latin1")

    # Tokenize words
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(files.data)

    # Transforms the occurrence counts to frequencies of word
    tf_transformer = TfidfTransformer()
    x_train_tf = tf_transformer.fit_transform(x)

    # Splits the data set for training and testing
    X_train_set, X_test_set, Y_train_set, Y_test_set = model_selection.train_test_split(x_train_tf, files.target,
                                                                                        train_size=0.8, test_size=0.20,
                                                                                        random_state=None)

    return files, X_train_set, X_test_set, Y_train_set, Y_test_set


def create_metrics(bayes_classifier, X_test_set, Y_test_set, files, prediction):
    disp = metric.plot_confusion_matrix(bayes_classifier, X_test_set, Y_test_set, display_labels=files.target_names,
                                        cmap=plt.cm.Blues, normalize="true")
    disp.ax_.set_title("Confusion matrix")
    print("b) \nConfusion matrix")
    print(disp.confusion_matrix)
    # plt.show()
    print("c) Precision, recall, F1 measure")
    print(metric.classification_report(Y_test_set, prediction, target_names=files.target_names))
    print("d)")
    print('Accuracy: ' + str(metric.accuracy_score(Y_test_set, prediction)))
    print('Macro-average F1: ' + str(metric.f1_score(Y_test_set, prediction, average='macro')))
    print('Weighted-average F1: ' + str(metric.f1_score(Y_test_set, prediction, average='weighted')))
    print("e)")
    dist = get_distribution()
    # print(dist)
    num_files = sum(len(files) for _, _, files in os.walk('BBC')) - 1

    for key in dist:
        print(key + ": " + str(dist[key] / num_files))

    print("f) " + str(bayes_classifier.n_features_))



def main():
    plot_distribution()
    files, X_train_set, X_test_set, Y_train_set, Y_test_set = prepare_datasets()

    bayes_classifier = MultinomialNB(alpha=0.0001).fit(X_train_set, Y_train_set)
    prediction = bayes_classifier.predict(X_test_set)
    create_metrics(bayes_classifier, X_test_set, Y_test_set, files, prediction)




if __name__ == "__main__":
    main()
