"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        n,d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        unique, counts = np.unique(y, return_counts=True)
        occ = dict(zip(unique, counts))
        self.probabilities_ = [
            occ[0]/float(y.size),
            occ[1]/float(y.size)
        ]
        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        y = np.random.choice([0, 1], size=len(X), replace=True, p=self.probabilities_)

        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)

    total_train_error = 0
    total_test_error = 0

    for i in range(1, 100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        clf.fit(X_train, y_train)      # fit training data using the classifier
        y_pred_train = clf.predict(X_train)        # take the classifier and run it on the training data
        total_train_error += 1 - metrics.accuracy_score(y_train, y_pred_train, normalize=True)
        y_pred_test = clf.predict(X_test)
        total_test_error += 1 - metrics.accuracy_score(y_test, y_pred_test, normalize=True)

    average_train_error = total_train_error / 100
    average_test_error = total_test_error / 100

    ### ========== TODO : END ========== ###

    return average_train_error, average_test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    #for i in range(d) :
        # plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)


    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)



    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    clf = RandomClassifier() # create Random classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print('Classifying using Decision Tree...')
    clf = DecisionTreeClassifier()
    clf = clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors
    print('Classifying using k-Nearest Neighbors...')
    clf = KNeighborsClassifier(n_neighbors=3)
    clf = clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error for k=3: %.3f' % train_error)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf = clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error for k=5: %.3f' % train_error)

    clf = KNeighborsClassifier(n_neighbors=7)
    clf = clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error for k=7: %.3f' % train_error)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    train_error, test_error = error(clf, X, y)
    print('\t-- MajorityVote: training error = %.3f, test error = %.3f' % (train_error , test_error))

    clf = RandomClassifier() # create Random classifier, which includes all model parameters
    train_error, test_error = error(clf, X, y)
    print('\t-- Random:       training error = %.3f, test error = %.3f' % (train_error , test_error))

    clf = DecisionTreeClassifier() # create DecisionTree classifier, which includes all model parameters
    train_error, test_error = error(clf, X, y)
    print('\t-- DecisionTree: training error = %.3f, test error = %.3f' % (train_error , test_error))

    clf = KNeighborsClassifier(n_neighbors=5) # create KNN classifier, which includes all model parameters
    train_error, test_error = error(clf, X, y)
    print('\t-- KNeighbors:   training error = %.3f, test error = %.3f' % (train_error , test_error))

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    kvals = []
    test_errors = []
    for i in range(1, 51):
        if i % 2 == 0:
            continue
        clf = KNeighborsClassifier(n_neighbors=i) # create KNN classifier
        kvals.append(i)
        test_errors.append(1 - np.average(cross_val_score(clf, X, y, cv=10)))
    plt.xlabel('k')
    plt.ylabel('10-fold cross validation error rate')
    print('\t-- the value of k that minimizes cross validation error is %d' % kvals[test_errors.index(min(test_errors))])
    # plt.plot(kvals, test_errors, marker='o')
    # plt.show()
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    depths = []
    train_errors = []
    test_errors = []
    for i in range(1, 21):
        clf = DecisionTreeClassifier(max_depth=i) # create depth classifier
        clf.fit(X, y)                  # fit training data using the classifier
        y_pred = clf.predict(X)        # take the classifier and run it on the training data
        train_errors.append(1 - metrics.accuracy_score(y, y_pred, normalize=True))
        depths.append(i)
        test_errors.append(1 - np.average(cross_val_score(clf, X, y, cv=10)))
        # print('for depth %d, the error is %.3f' % (i, 1 - np.average(cross_val_score(clf, X, y, cv=10))))
    plt.xlabel('depth')
    plt.ylabel('10-fold cross validation error rate')
    print('\t-- the depth that minimizes cross validation error is %d' % kvals[test_errors.index(min(test_errors))])
    # plt.plot(depths, test_errors, marker='o', label='Test Error')
    # plt.plot(depths, train_errors, marker='x', label='Training Error')
    # plt.legend(loc='lower left')
    # plt.show()

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    splits = []
    all_errors = {
        'knn_train_errors' : [],
        'knn_test_errors'  : [],
        'dt_train_errors'  : [],
        'dt_test_errors'   : []
    }
    for i in range(1, 11):
        if i == 10:
            X_train_subset = X_train
            y_train_subset = y_train
        else:
            X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(X_train, y_train, train_size=i*0.1)
        splits.append(i*0.1)


        errors = {}
        clf = KNeighborsClassifier(n_neighbors=7) # create KNN classifier, which includes all model parameters
        clf.fit(X_train_subset, y_train_subset)      # fit training data using the classifier
        y_pred_train = clf.predict(X_train_subset)        # take the classifier and run it on the training data
        knn_train_error = 1 - metrics.accuracy_score(y_train_subset, y_pred_train, normalize=True)
        y_pred_test = clf.predict(X_test)
        knn_test_error = 1 - metrics.accuracy_score(y_test, y_pred_test, normalize=True)
        # print('knn with %.1f training had %.3f training error and %.3f test error' % (0.1*i, knn_train_error, knn_test_error))

        clf = DecisionTreeClassifier(max_depth=11) # create DecisionTree classifier, which includes all model parameters
        clf.fit(X_train_subset, y_train_subset)      # fit training data using the classifier
        y_pred_train = clf.predict(X_train_subset)        # take the classifier and run it on the training data
        dt_train_error = 1 - metrics.accuracy_score(y_train_subset, y_pred_train, normalize=True)
        y_pred_test = clf.predict(X_test)
        dt_test_error = 1 - metrics.accuracy_score(y_test, y_pred_test, normalize=True)
        # print('dt  with %.1f training had %.3f training error and %.3f test error' % (0.1*i, knn_train_error, knn_test_error))

        all_errors['knn_train_errors'].append(knn_train_error)
        all_errors['knn_test_errors'].append(knn_test_error)
        all_errors['dt_train_errors'].append(dt_train_error)
        all_errors['dt_test_errors'].append(dt_test_error)

    plt.title('KNeighbors Learning Curves')
    plt.xlabel('portion of training set used')
    plt.ylabel('error rate')
    plt.plot(splits, all_errors['knn_train_errors'], marker='o', label='Training Error')
    plt.plot(splits, all_errors['knn_test_errors'], marker='x', label='Test Error')
    plt.legend(loc='upper right')
    plt.show()

    plt.title('Decision Tree Learning Curves')
    plt.xlabel('portion of training set used')
    plt.ylabel('error rate')
    plt.plot(splits, all_errors['dt_train_errors'], marker='o', label='Training Error')
    plt.plot(splits, all_errors['dt_test_errors'], marker='x', label='Test Error')
    plt.legend(loc='upper right')
    plt.show()
    ### ========== TODO : END ========== ###


    print('Done')


if __name__ == "__main__":
    main()
