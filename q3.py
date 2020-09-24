import argparse
import numpy as np
import pandas as pd
import copy
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def get_param(classifier, xFeat, y, xTest, yTest):
    if classifier == "knn":
        clf = GridSearchCV(
            KNeighborsClassifier(),
            {'n_neighbors': range(1, 50, 1)},
            cv=5, scoring='f1_macro')
        clf.fit(xFeat, y['label'])
    else:
        clf = GridSearchCV(
            DecisionTreeClassifier(),
            [{'max_depth': range(1, 20),
              'min_samples_leaf': range(1, 30),
              'criterion': ['entropy', 'gini']}],
            cv=5, scoring='f1_macro')
        clf.fit(xFeat, y)

    optimal_parameter = clf.best_params_
    optimal_parameter_string = str(optimal_parameter)

    print("optimal parameter: " + optimal_parameter_string)

    return optimal_parameter


def auc(model, classifier, xFeat, y, xTest, yTest):
    model.fit(xFeat, y['label'])
    predict_probability = model.predict_proba(xTest)
    fpr, tpr, thresh = metrics.roc_curve(yTest, predict_probability[:, 1])
    auc_calc = metrics.auc(fpr, tpr)

    if classifier == 'knn':
        print("knn auc: " + str(round(auc_calc, 5)))
    else:
        print("decision tree auc: " + str(round(auc_calc, 5)))
    return auc_calc


def acc(model, classifier, xFeat, y, xTest, yTest):
    model.fit(xFeat, y['label'])
    predict_y = model.predict(xTest)
    accuracy = accuracy_score(yTest['label'], predict_y)

    if classifier == 'knn':
        print("knn percent accuracy: " + str(round(accuracy, 5)))
    else:
        print("decision tree percent accuracy: " + str(round(accuracy, 5)))
    return accuracy


def drop(percent, xFeat, y):
    xFeat_copy = copy.deepcopy(xFeat)
    y_copy = copy.deepcopy(y)

    # rain_drop is what we want to remove
    rain_drop = list(range(0, len(xFeat_copy)))
    np.random.shuffle(rain_drop)
    # drop_top is the total we want to drop
    drop_top = round(percent * len(xFeat_copy))

    xTrain = xFeat_copy.drop(rain_drop[0:drop_top])
    yTrain = y_copy.drop(rain_drop[0:drop_top])

    return xTrain, yTrain


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    # create dt and knn objects
    knn_param = get_param("knn", xTrain, yTrain, xTest, yTest)
    dt_param = get_param("dt", xTrain, yTrain, xTest, yTest)

    # 5%
    xTrain_5_percent, yTrain_5_percent = drop(0.05, xTrain, yTrain)
    # 10%
    xTrain_10_percent, yTrain_10_percent = drop(0.1, xTrain, yTrain)
    # 20%
    xTrain_20_percent, yTrain_20_percent = drop(0.2, xTrain, yTrain)

    # knn
    knn = KNeighborsClassifier(
        n_neighbors=knn_param["n_neighbors"])

    # auc and acc for knn
    print("Overall:")
    auc_knn = auc(knn, "knn", xTrain, yTrain, xTest, yTest)
    accuracy_knn = acc(knn, "knn", xTrain, yTrain, xTest, yTest)
    print("For 5%:")
    auc_knn_5_percent = auc(knn, "knn", xTrain_5_percent, yTrain_5_percent, xTest, yTest)
    acc_knn_5_percent = acc(knn, "knn", xTrain_5_percent, yTrain_5_percent, xTest, yTest)
    print("For 10%:")
    auc_knn_10_percent = auc(knn, "knn", xTrain_10_percent, yTrain_10_percent, xTest, yTest)
    acc_knn_10_percent = acc(knn, "knn", xTrain_10_percent, yTrain_10_percent, xTest, yTest)
    print("For 20%:")
    auc_knn_20_percent = auc(knn, "knn", xTrain_20_percent, yTrain_20_percent, xTest, yTest)
    acc_knn_20_percent = acc(knn, "knn", xTrain_20_percent, yTrain_20_percent, xTest, yTest)

    # decision tree
    dt = DecisionTreeClassifier(
        criterion=dt_param['criterion'],
        max_depth=dt_param['max_depth'],
        min_samples_leaf=dt_param['min_samples_leaf'])

    # auc and acc for dt
    print("Overall:")
    auc_dt = auc(dt, "dt", xTrain, yTrain, xTest, yTest)
    acc_dt = acc(dt, "dt", xTrain, yTrain, xTest, yTest)
    print("For 5%:")
    auc_dt_5_percent = auc(dt, "dt", xTrain_5_percent, yTrain_5_percent, xTest, yTest)
    acc_dt_5_percent = acc(dt, "dt", xTrain_5_percent, yTrain_5_percent, xTest, yTest)
    print("For 10%:")
    auc_dt_10_percent = auc(dt, "dt", xTrain_10_percent, yTrain_10_percent, xTest, yTest)
    acc_dt_10_percent = acc(dt, "dt", xTrain_10_percent, yTrain_10_percent, xTest, yTest)
    print("For 20%:")
    auc_dt_20_percent = auc(dt, "dt", xTrain_20_percent, yTrain_20_percent, xTest, yTest)
    acc_dt_20_percent = acc(dt, "dt", xTrain_20_percent, yTrain_20_percent, xTest, yTest)


if __name__ == "__main__":
    main()
