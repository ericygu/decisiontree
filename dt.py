import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
import math


class Node(object):
    def __init__(self, matrix, array, split=None, left=None, right=None):
        self.matrix = matrix
        self.array = array
        self.split = split
        self.left = left
        self.right = right


def get_gini(array):
    try:
        labels = array
        probability_one = sum(labels) / len(labels)
        probability_two = 1 - probability_one
        gini_one = probability_one * math.log2(probability_one)
        gini_two = probability_two * math.log2(probability_two)
        gini = gini_one + gini_two
        return gini
    except:
        return 0


def get_entropy(array):
    try:
        labels = array
        probability_one = sum(labels) / len(labels)
        probability_two = 1 - probability_one
        entropy = probability_one * math.log2(probability_two)
        return entropy
    except:
        return 0


class DecisionTree(object):
    maxDepth = 0  # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None  # splitting criterion

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.criterion = criterion
        if self.criterion == "entropy":
            self.criterion = get_entropy
        elif self.criterion == "gini":
            self.criterion = get_gini

    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """

        def splitter(mat, arr, feature, split_factor):
            # feature and split_factor should both be some numerical value

            split_a = mat[mat[:, feature] >= split_factor]
            split_b = arr[mat[:, feature] >= split_factor]

            split_c = mat[mat[:, feature] < split_factor]
            split_d = arr[mat[:, feature] < split_factor]

            return [split_a, split_b], [split_c, split_d]

        def get_split(mat, arr):

            matrix_iter = range(len(mat[0]))
            cri = self.criterion
            feat = 0
            val = mat[0][0]

            split_a, split_b = splitter(mat, arr, feat, val)

            a_one = split_a[1]
            b_one = split_b[1]

            split_value_a = (len(a_one) / len(arr)) * cri(a_one)
            split_value_b = (len(b_one) / len(arr)) * cri(b_one)

            split_value = split_value_a + split_value_b

            for feature in matrix_iter:
                feature_and_matrix = mat[:, feature]
                for value in feature_and_matrix:

                    min_leaf = self.minLeafSample

                    split_c, split_d = splitter(mat, arr, feature, value)

                    c_one = split_c[1]
                    d_one = split_d[1]

                    if not len(c_one) >= min_leaf:
                        continue
                    elif not len(d_one) >= min_leaf:
                        continue
                    else:
                        cri_value_c = (len(c_one) / len(arr)) * cri(c_one)
                        cri_value_d = (len(d_one) / len(arr)) * cri(d_one)

                        cri_value = cri_value_c + cri_value_d

                        if cri_value < split_value:
                            split_a = split_c
                            split_b = split_d
                            feat = feature
                            val = value
                            split_value = cri_value

            return split_a, split_b, [feat, val]

        def recursion_tree(node, tree_depth):
            max_depth = self.maxDepth
            node_array = node.array
            node_matrix = node.matrix

            if tree_depth >= max_depth:
                return node, tree_depth
            elif sum(node_array) == len(node_array):
                return node, tree_depth
            elif sum(node_array) == 0:
                return node, tree_depth
            elif len(node_matrix[0]) < 1:
                return node, tree_depth
            else:

                left, right, split = get_split(node_matrix, node_array)
                min_leaf = self.minLeafSample

                if len(left[1]) < min_leaf:
                    return node, tree_depth
                elif len(right[1]) < min_leaf:
                    return node, tree_depth
                else:
                    node.split = split
                    node.left = Node(left[0], left[1])
                    node.right = Node(right[0], right[1])

                    tree_depth += 1
                    recursion_tree(node.left, tree_depth)
                    recursion_tree(node.right, tree_depth)

        xFeat_numpy = xFeat.to_numpy()
        y_numpy = y.to_numpy()

        self.root = Node(xFeat_numpy, y_numpy)
        recursion_tree(self.root, 0)

        return self

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """
        x_feature = xFeat.to_numpy()
        yHat = []  # variable to store the estimated class label

        def check(array):
            labels = array
            try:
                if sum(labels) / len(labels) >= .5:
                    return 1
                else:
                    return 0
            except len(labels) == 0:
                return 0

        for rows in x_feature:
            node = self.root

            while node.split is not None:
                feature, value = node.split

                if rows[feature] >= value:
                    node = node.left
                else:
                    node = node.right

            yHat.append(check(node.array))

        return yHat


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
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
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
