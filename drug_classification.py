import sys

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from sklearn.naive_bayes import GaussianNB
import sklearn.tree as tree
import sklearn.metrics as metric
from pandas.api.types import CategoricalDtype

def plot_distribution(drugs):
    drug_distribution = {"drugA": 0, "drugB": 0, "drugC": 0, "drugX": 0, "drugY": 0}

    for i in drugs.index:
        drug_distribution[drugs["Drug"][i]] += 1

    plt.bar(list(drug_distribution.keys()), drug_distribution.values())
    # plt.show()
    plt.savefig("drug-distribution.pdf")


def main():
    drugs = pd.read_csv(r'drug200.csv')

    plot_distribution(drugs)

    drugs_dtype = CategoricalDtype(categories=['drugA','drugB','drugC','drugX','drugY'], ordered=False)
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

    test = pd.get_dummies(drugs)
    print(test)

    X_train_set, X_test_set, Y_train_set, Y_test_set = model_selection.train_test_split(table, encoded_drugs)


    #print(X_test_set)
    # print("X_TEST_SET")
    # print(Y_train_set)
    # print("888888888888888888888888888888888")
    # print(Y_train_set)
    # print("888888888888888888888888888888888")
    # print(Y_test_set)

    # gaussianNB = GaussianNB().fit(X_train_set, Y_train_set)
    # gaussianNB_prediction = gaussianNB.predict(X_test_set)
    # print(metric.accuracy_score(Y_test_set, gaussianNB_prediction))

    # baseDt = tree.DecisionTreeClassifier().fit(X_train_set, Y_train_set)
    # baseDt_prediction = baseDt.predict(X_test_set)
    # print(metric.accuracy_score(Y_test_set, baseDt_prediction))


if __name__ == "__main__":
    main()
