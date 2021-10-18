import matplotlib.pyplot as plt
import os, os.path
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.model_selection as model_selection
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as metric

sum_words_frequency1 = 0

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

    # print(vectorizer.vocabulary_.get(u'sport'))
    # print(vectorizer.vocabulary_.get(u'super'))

    # Transforms the occurrence counts to frequencies of word
    tf_transformer = TfidfTransformer()
    x_train_tf = tf_transformer.fit_transform(x)

    # Splits the data set for training and testing
    X_train_set, X_test_set, Y_train_set, Y_test_set = model_selection.train_test_split(x_train_tf, files.target,
                                                                                        train_size=0.8, test_size=0.20,
                                                                                        random_state=None)

    get_frequency_of_one(x)


    return files, X_train_set, X_test_set, Y_train_set, Y_test_set


def create_metrics(bayes_classifier, X_test_set, Y_test_set, files, prediction):
    # disp = metric.plot_confusion_matrix(bayes_classifier, X_test_set, Y_test_set, display_labels=files.target_names,
    #                                     cmap=plt.cm.Blues, normalize="true")
    # disp.ax_.set_title("Confusion matrix")
    # print("b) \nConfusion matrix")
    # print(disp.confusion_matrix)
    # # plt.show()

    # cm = metric.confusion_matrix(Y_test_set, prediction, labels=files.target_names)
    disp = metric.ConfusionMatrixDisplay.from_predictions(Y_test_set, prediction, display_labels=files.target_names,
                                                          normalize='true')
    print("b) \nConfusion matrix")
    disp.plot()
    plt.savefig("confusion-matrix.pdf")
    print("NOTE: I had difficulty making this work because the function in the assignment handout is deprecated as in "
          "python 3.9, \nwhich is the version on my computer. Instead, the confusion matrix is on the file called "
          "confusion-matrix.pdf")
    # print(plt.show())
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

    print("f) " + str(bayes_classifier.n_features_in_))
    print("g)")
    tokens_bussiness = get_word_tokens('BBC/business')
    tokens_ent = get_word_tokens('BBC/entertainment')
    tokens_pol = get_word_tokens('BBC/politics')
    tokens_sport = get_word_tokens('BBC/sport')
    tokens_tech = get_word_tokens('BBC/tech')

    print("Businness: " + str(tokens_bussiness))
    print("Entertainment: " + str(tokens_ent))
    print("Politics: " + str(tokens_pol))
    print("Sport: " + str(tokens_sport))
    print("Tech: " + str(tokens_tech))

    print("h)")
    print(tokens_bussiness + tokens_tech + tokens_pol + tokens_sport + tokens_ent)

    print("i)")
    bus_0, bus_per = get_frequency_of_zero(bayes_classifier, 0)
    ent_0, ent_per = get_frequency_of_zero(bayes_classifier,1)
    pol_0, pol_per = get_frequency_of_zero(bayes_classifier, 2)
    sport_0, sport_per = get_frequency_of_zero(bayes_classifier, 3)
    tech_0, tech_per = get_frequency_of_zero(bayes_classifier, 4)

    print("Business: " + str(bus_0) + ", " + str(bus_per) + "%")
    print("Entertainment: " + str(ent_0) + ", " + str(ent_per) + "%")
    print("Politics: " + str(pol_0) + ", " + str(pol_per) + "%")
    print("Sport: " + str(sport_0) + ", " + str(sport_per) + "%")
    print("Tech: " + str(tech_0) + ", " + str(tech_per) + "%")
    print("j)")
    print(sum_words_frequency1)
    print("k)")
    print("Sport: " + str(bayes_classifier.feature_log_prob_[1][24932]))
    print("Super: " + str(bayes_classifier.feature_log_prob_[4][25704]))


def get_word_tokens(path):
    import re
    count = 0

    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r') as f:
            for line in f:
                count += len(re.findall('[^\d\W]+', line))

    return count

def get_frequency_of_one(vocab):

    test = vocab.toarray().sum(axis=0)
    global sum_words_frequency1
    for i in test:
        if i == 1:
            sum_words_frequency1 += 1

def get_frequency_of_zero(bayes, number):

    sum = 0
    zero = 0
    for i in bayes.feature_count_[number]:
        sum += 1
        if i == 0:
            zero += 1

    return zero, zero/sum

def main():
    plot_distribution()
    files, X_train_set, X_test_set, Y_train_set, Y_test_set = prepare_datasets()

    bayes_classifier = MultinomialNB().fit(X_train_set, Y_train_set)
    prediction = bayes_classifier.predict(X_test_set)
    create_metrics(bayes_classifier, X_test_set, Y_test_set, files, prediction)


if __name__ == "__main__":
    main()
