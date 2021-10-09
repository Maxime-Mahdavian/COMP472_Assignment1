import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from sklearn.naive_bayes import GaussianNB
import sklearn.tree as tree
import sklearn.metrics as metric
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

i = 1
accuracy = list()
macroF1 = list()
weightedF1 = list()

def plot_distribution(drugs):
    drug_distribution = {"drugA": 0, "drugB": 0, "drugC": 0, "drugX": 0, "drugY": 0}

    for i in drugs.index:
        drug_distribution[drugs["Drug"][i]] += 1

    plt.bar(list(drug_distribution.keys()), drug_distribution.values())
    # plt.show()
    plt.savefig("drug-distribution.pdf")


def create_metrics(clf, X_train, X_test, Y_train, Y_test, prediction):
    target = ('drugA', 'drugB', 'drugC', 'drugX', 'drugY')
    global i
    name = 'confusion-matrix' + str(i) + ".pdf"
    i += 1
    disp = metric.ConfusionMatrixDisplay.from_predictions(Y_test, prediction, display_labels=target,
                                                          normalize='true')
    print("b) \nConfusion matrix")
    disp.plot()
    # plt.savefig(name)
    # print(plt.show())
    print("c) Precision, recall, F1 measure")
    print(metric.classification_report(Y_test, prediction, target_names=target))
    print("d)")
    print('Accuracy: ' + str(metric.accuracy_score(Y_test, prediction)))
    accuracy.append(metric.accuracy_score(Y_test, prediction))
    print('Macro-average F1: ' + str(metric.f1_score(Y_test, prediction, average='macro')))
    macroF1.append(metric.f1_score(Y_test, prediction, average='macro'))
    print('Weighted-average F1: ' + str(metric.f1_score(Y_test, prediction, average='weighted')))
    weightedF1.append(metric.f1_score(Y_test, prediction, average='weighted'))
    print("-------------------------------------------------------------------------------------")


def main():
    drugs = pd.read_csv(r'drug200.csv')
    plot_distribution(drugs)

    drugs_dtype = CategoricalDtype(categories=['drugA', 'drugB', 'drugC', 'drugX', 'drugY'], ordered=False)
    encoded_drugs = drugs.Drug.astype(drugs_dtype).cat.codes
    encoded_sex = drugs.Sex.astype("category").cat.codes
    bp_dtype = CategoricalDtype(categories=['LOW', 'NORMAL', 'HIGH'], ordered=True)
    encoded_bp = drugs.BP.astype(bp_dtype).cat.codes
    chol_dtype = CategoricalDtype(categories=['NORMAL', 'HIGH'], ordered=True)
    encoded_chol = drugs.Cholesterol.astype(chol_dtype).cat.codes

    table = pd.DataFrame(
        list(zip(drugs['Age'], encoded_sex, encoded_bp, encoded_chol, drugs['Na_to_K'], encoded_drugs)),
        columns=['Age', 'sex', 'BP', 'Cholestoerol', 'Na_to_K', 'Drug'])
    # print(table)

    target = table.iloc[:, 5]
    attributes = table.iloc[:, 0:5]

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(attributes, target)

    for x in range(10):
        gaussianNB = GaussianNB().fit(X_train, Y_train)
        gaussianNB_prediction = gaussianNB.predict(X_test)
        print("Gaussian")
        create_metrics(gaussianNB, X_train, X_test, Y_train, Y_test, gaussianNB_prediction)

        baseDt = tree.DecisionTreeClassifier().fit(X_train, Y_train)
        baseDt_prediction = baseDt.predict(X_test)
        print("Base-DT")
        create_metrics(baseDt, X_train, X_test, Y_train, Y_test, baseDt_prediction)

        dt_parameters = {'criterion': ['gini', 'entropy'], 'max_depth': [4, 6], 'min_samples_split': [3, 5, 7]}
        grid = GridSearchCV(baseDt, dt_parameters)
        grid = grid.fit(X_train, Y_train)
        grid_prediction = grid.predict(X_test)
        print("Top-DT")
        print("Best parameters")
        print(grid.best_params_)
        # print(metric.accuracy_score(Y_test, grid_prediction))
        create_metrics(grid, X_train, X_test, Y_train, Y_test, grid_prediction)

        per = Perceptron().fit(X_train, Y_train)
        per_prediction = per.predict(X_test)
        print("Perceptron")
        # print(metric.accuracy_score(Y_test, per_prediction))
        create_metrics(per, X_train, X_test, Y_train, Y_test, per_prediction)

        mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(100,), solver='sgd').fit(X_train, Y_train)
        mlp = MLPClassifier().fit(X_train, Y_train)
        mlp_prediction = mlp.predict(X_test)
        print("MLP")
        # print(metric.accuracy_score(Y_test, mlp_prediction))
        create_metrics(mlp, X_train, X_test, Y_train, Y_test, mlp_prediction)

        mlp_parameters = {'activation': ['logistic', 'tanh', 'relu', 'identity'],
                          'hidden_layer_sizes': [(30, 50), (10, 10, 10)], 'solver': ['adam', 'sgd']}
        mlp_grid = GridSearchCV(mlp, mlp_parameters, return_train_score=True)
        mlp_grid = mlp_grid.fit(X_train, Y_train)
        mlp_grid_prediction = mlp_grid.predict(X_test)
        print("Top-MLP")
        print("Best parameters")
        print(mlp_grid.best_params_)
        # print(metric.accuracy_score(Y_test, mlp_grid_prediction))
        create_metrics(mlp_grid, X_train, X_test, Y_train, Y_test, mlp_grid_prediction)

    print("\nRESULTS")
    print("Average accuracy")
    print(np.average(accuracy))
    print("Average macro F1")
    print(np.average(macroF1))
    print("Average weighted F1")
    print(np.average(weightedF1))
    print("Accuracy standard deviation")
    print(np.std(accuracy))
    print("Macro F1 standard deviation")
    print(np.std(macroF1))
    print("Weighted F1 standard deviation")
    print(np.std(weightedF1))

    # mlp_test = MLPClassifier(activation='tanh', hidden_layer_sizes=(30,50), solver='adam').fit(X_train, Y_train)
    # test_mlp_test = mlp_test.predict(X_test)
    # create_metrics(mlp_test, X_train, X_test, Y_train, Y_test, test_mlp_test)

if __name__ == "__main__":
    main()
