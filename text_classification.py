import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import sklearn as sk
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.model_selection as model_selection
from sklearn.naive_bayes import MultinomialNB


def plot_distribution():
    dist = {}

    for folder in os.listdir('BBC'):
        if os.path.isfile(os.path.join('BBC', folder)):
            continue
        else:
            dist[folder] = len([name for name in os.listdir('BBC/' + folder)])

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
    # print(x_train_tf.shape)

    # Splits the data set for training and testing
    train_set, test_set = model_selection.train_test_split(x_train_tf, train_size=0.8, test_size=0.20,
                                                           random_state=None)
    return files, train_set, test_set


def main():
    plot_distribution()
    # files, train_set, test_set = prepare_datasets()
    # print(train_set.shape)
    # print(test_set.shape)

    files = load_files('BBC', encoding="latin1")

    # Tokenize words
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(files.data)

    # Transforms the occurrence counts to frequencies of word
    tf_transformer = TfidfTransformer()
    x_train_tf = tf_transformer.fit_transform(x)
    # print(x_train_tf.shape)

    # Splits the data set for training and testing
    train_set, test_set = model_selection.train_test_split(x, train_size=0.8,
                                                           test_size=0.20,
                                                        random_state=None)
    y = np.zeros(int(x.shape[0] * 0.8))
    clf = MultinomialNB()
    clf.fit(train_set, y)

    prediction = clf.predict(test_set)
    print(prediction)

if __name__ == "__main__":
    main()
