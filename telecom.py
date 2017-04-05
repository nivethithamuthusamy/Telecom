from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
#%matplotlib inline

#read data
churn_df = pd.read_csv('C:/Users/Nivi/Documents/Python/Telecom/Telecom_data.csv')
col_names = churn_df.columns.tolist()

print("Column names:")
print(col_names)

to_show = col_names[:6] + col_names[-6:]

print("\nSample data:")
churn_df[to_show].head(6)

# The shape of the dataset
print (churn_df.shape)


# types
pd.set_option('display.max_rows', 500)
print(churn_df.dtypes)


# descriptions, change precision to 3 places
pd.set_option('precision', 3)
print(churn_df.describe())

# histograms
churn_df.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
plt.show()

# density
churn_df.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=False,
fontsize=1)
plt.show()

# box and whisker plots
churn_df.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False,
fontsize=1)
plt.show()

# correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(churn_df.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(col_names)
ax.set_yticklabels(col_names)
plt.show()

# Isolate target data
churn_result = churn_df['Churn?']
y = np.where(churn_result == 'True.',1,0)


# We don't need these columns
to_drop = ['State','Area Code','Phone','Churn?']
churn_feat_space = churn_df.drop(to_drop,axis=1)

# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

#Imbalance in the class data, we will handle this thorugh Kfold or cross validation

print('There are {} instances for churn class and {} instances for not-churn classes.'.format(y.sum(), y.shape[0] - y.sum()))


# Pull out features for future use
features = churn_feat_space.columns
print(features)

X = churn_feat_space.as_matrix().astype(np.float)

# This is important standardization of data

scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Feature space holds %d observations and %d features" % X.shape)
print("Unique target labels:", np.unique(y))

from sklearn.cross_validation import KFold

def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=3,shuffle=True)
    y_pred = y.copy()
    
    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred
    
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import average_precision_score

def accuracy(y_true,y_pred):
    # NumPy interpretes True and False as 1. and 0.
    return np.mean(y_true == y_pred)

print("Logistic Regression:")
print("%.3f" % accuracy(y, run_cv(X,y,LR)))
print("Gradient Boosting Classifier")
print("%.3f" % accuracy(y, run_cv(X,y,GBC)))
print("Support vector machines:")
print("%.3f" % accuracy(y, run_cv(X,y,SVC)))
print("Random forest:")
print("%.3f" % accuracy(y, run_cv(X,y,RF)))
print("K-nearest-neighbors:")
print("%.3f" % accuracy(y, run_cv(X,y,KNN)))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns

def draw_confusion_matrices(confusion_matricies,class_names):
    class_names = class_names.tolist()
    for cm in confusion_matrices:
        classifier, cm = cm[0], cm[1]
        print(cm)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix for %s' % classifier)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
y = np.array(y)
class_names = np.unique(y)


confusion_matrices = {

                1: {
                    'matrix': confusion_matrix(y,run_cv(X,y,SVC)),
                    'title': 'Support Vector Machine',
                   },
                2: {
                    'matrix': confusion_matrix(y,run_cv(X,y,RF)),
                    'title': 'Random Forest',
                   },
                3: {
                    'matrix': confusion_matrix(y,run_cv(X,y,KNN)),
                    'title': 'K Nearest Neighbors',
                   },
                4: {
                    'matrix': confusion_matrix(y,run_cv(X,y,LR)),
                    'title': 'Logistic Regression',
                   },
                5: {
                    'matrix': confusion_matrix(y,run_cv(X,y,GBC)),
                    'title': 'Gradient Boosting Classifier',
                   },
}



fix, ax = plt.subplots(figsize=(16, 12))
plt.suptitle('Confusion Matrix of Various Classifiers')
for ii, values in confusion_matrices.items():
    matrix = values['matrix']
    title = values['title']
    plt.subplot(3, 3, ii) # starts from 1
    plt.title(title);
    sns.heatmap(matrix, annot=True,  fmt='');
    

print("Support Vector Machines F1 Score" ,f1_score(y,run_cv(X,y,SVC)))
print("Random Forest F1 Score" ,f1_score(y,run_cv(X,y,RF)))
print("K-Nearest-Neighbors F1 Score" ,f1_score(y,run_cv(X,y,KNN)))
print("Gradient Boosting Classifier F1 Score" ,f1_score(y,run_cv(X,y,GBC)))
print("Logisitic Regression" ,f1_score(y,run_cv(X,y,LR)))


train_index,test_index = train_test_split(churn_df.index)

forest = RF()
forest_fit = forest.fit(X[train_index], y[train_index])
forest_predictions = forest_fit.predict(X[test_index])


importances = forest_fit.feature_importances_[:10]
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))

# Plot the feature importances of the forest
#import pylab as pl
plt.figure()
plt.title("Feature importances")
plt.bar(range(10), importances[indices], yerr=std[indices], color="r", align="center")
plt.xticks(range(10), indices)
plt.xlim([-1, 10])
plt.show()


def run_prob_cv(X, y, clf_class, roc=False, **kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y),2))
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob
    
import warnings
warnings.filterwarnings('ignore')

# Use 10 estimators so predictions are all multiples of 0.1
pred_prob = run_prob_cv(X, y, RF, n_estimators=10)
pred_churn = pred_prob[:,1]
is_churn = y == 1

# Number of times a predicted probability is assigned to an observation
counts = pd.value_counts(pred_churn)
counts[:]

from collections import defaultdict
true_prob = defaultdict(float)

# calculate true probabilities
for prob in counts.index:
    true_prob[prob] = np.mean(is_churn[pred_churn == prob])
true_prob = pd.Series(true_prob)

# pandas-fu
counts = pd.concat([counts,true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
counts

#Implementation of RF

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


forest = RF()
forest_fit = forest.fit(X[train_index], y[train_index])
predictions = forest_fit.predict(X[test_index])

#In case like to import predictions in CSV file
#pred_prob = forest_fit.predict_proba(X[test_index])
#final_pred = pd.DataFrame(forest_predictions, columns=['predictions'])#.to_csv('prediction.csv')


print(confusion_matrix(y[test_index],predictions))
print(accuracy_score(y[test_index],predictions))
print(classification_report(y[test_index],predictions))

#Final Implementation of RF with best parameters after grid search

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


forest = RF(n_estimators=700, max_features= 'auto')
forest_fit = forest.fit(X[train_index], y[train_index])
predictions = forest_fit.predict(X[test_index])

#In case like to import predictions in CSV file
#pred_prob = forest_fit.predict_proba(X[test_index])
#final_pred = pd.DataFrame(forest_predictions, columns=['predictions'])#.to_csv('prediction.csv')


#print(confusion_matrix(y[test_index],predictions))
print(accuracy_score(y[test_index],predictions))
print(classification_report(y[test_index],predictions))

print( "Random forests senstivity analysis Train Data:")
plot_roc(X[train_index],y[train_index],RF,n_estimators=700, max_features= 'auto')
print("Random forests senstivity analysis Test Data:")
plot_roc(X[test_index],y[test_index],RF,n_estimators=700, max_features= 'auto')

import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


RANDOM_STATE = 123


# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for paralellised ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 5
max_estimators = 700

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()