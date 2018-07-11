'''
done in  1892.08915114  seconds
Selected Features, Scores, P-Values
[('exercised_stock_options', '27.45', '0.000'), ('total_stock_value', '23.67', '0.000'), ('bonus', '15.80', '0.000'), ('salary', '10.90', '0.001'), ('deferred_income', '10.29', '0.002'), ('restricted_stock', '8.46', '0.004'), ('total_payments', '8.41', '0.004'), ('long_term_incentive', '8.36', '0.004'), ('shared_receipt_with_poi', '7.48', '0.007'), ('from_poi_to_this_person', '4.28', '0.040'), ('other', '3.96', '0.049'), ('poi_messages_total_messages_ratio', '3.87', '0.051'), ('loan_advances', '3.85', '0.052')]
tester classification report:
['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'poi_messages_total_messages_ratio']
/Applications/anaconda/lib/python2.7/site-packages/sklearn/feature_selection/univariate_selection.py:113: UserWarning: Features [3] are constant.
  UserWarning)
/Applications/anaconda/lib/python2.7/site-packages/sklearn/feature_selection/univariate_selection.py:114: RuntimeWarning: invalid value encountered in divide
  f = msb / msw
Pipeline(memory=None,
     steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=False)), ('skb', SelectKBest(k=13, score_func=<function f_classif at 0x10d2717d0>)), ('pca', PCA(copy=True, iterated_power='auto', n_components=1, random_state=42,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='distance'))])
  Accuracy: 0.87080 Precision: 0.52120  Recall: 0.38100 F1: 0.44021 F2: 0.40266
  Total predictions: 15000  True positives:  762  False positives:  700 False negatives: 1238 True negatives: 12300

'''

#!/usr/bin/python

import sys
import pickle
from pandas import DataFrame
import numpy as np
from pprint import pprint
sys.path.append("../tools/")

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler,  MaxAbsScaler, Imputer
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, accuracy_score, precision_score, f1_score

from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr

import time

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from project_feature_selection import select_k_best

from sklearn.utils.multiclass import type_of_target


# Define various feature lists
finance_features = ['salary', 'deferral_payments', 'total_payments', \
                    'loan_advances', 'bonus', 'restricted_stock_deferred', \
                    'deferred_income', 'total_stock_value', 'expenses', \
                    'exercised_stock_options', 'other', 'long_term_incentive', \
                    'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', \
                  'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
email_address_feature = ['email_address']
label_feature = ['poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = DataFrame.from_dict(data_dict, orient='index')
print 'Number of rows in original Enron dataset: ', len(df)
print 'Number of features in original Enron dataset: ', len(df.columns)

### Data Wrangling
# reordering columns and not include email_address_feature as this feature will not help with identifying POIs
df = df[finance_features + email_features + label_feature]

# Replacing NaN with 0 in finacial features
for feature in finance_features:
    df[feature] = df[feature].astype(float)
#     print 'Number of NaN in ', feature, 'column: ', df[feature].isnull().sum()
    df[feature] = df[feature].fillna(df[feature].median())
# print 'All NaNs in finance features are replaced with 0'


# # Replacing NaNs in email features with 0
for feature in email_features:
    df[feature] = df[feature].astype(float)
#     print 'Number of NaN in ', feature, 'column: ', df[feature].isnull().sum()
    df[feature] = df[feature].fillna(df[feature].median())
#     df[feature] = df[feature].astype(int)
# print 'All NaNs in email features are replaced with 0'

# # Converting POI column to int so for POI it will show 1 and for non-POI it will show 0
# df[label_feature] = df[label_feature].astype(int)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Task 2: Remove outliers
### Task 3: Create new feature poi_messages_total_messages_ratio.

df = df.drop('TOTAL')
df = df.drop('THE TRAVEL AGENCY IN THE PARK')
print 'outliners dropped.'

poi_email_interaction = df['from_poi_to_this_person'] + df['from_this_person_to_poi']
total_email_interaction = df['to_messages'] + df['from_messages']
df['poi_messages_total_messages_ratio'] = poi_email_interaction / total_email_interaction
email_features += ['poi_messages_total_messages_ratio']

features_list = ['poi'] + finance_features + email_features
### Store to my_dataset for easy export below.
my_dataset = df.T.to_dict()


### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size = 0.3, random_state = 42, stratify = labels)

# pprint(select_k_best(features_train, labels_train, '10'))

#
pipe = Pipeline(
                [
                 ('scaler', StandardScaler()),
                 ('skb', SelectKBest(f_classif)),
                 ('pca', PCA(random_state=42)),
                 ('clf', KNeighborsClassifier())
                ]
               )
# TODO: n_split = 50 best score so far
sss = StratifiedShuffleSplit(n_splits = 100, random_state = 42)

# Set grid search params
grid_params = {
                 'scaler__with_std': [True, False],
                 'skb__k': range(5, 15),
                 'pca__n_components': range(1,5),
                 'clf__weights': ['distance', 'uniform'],
                 'clf__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                 'clf__n_neighbors': [3,5,10]
                }



# Construct grid search
gs = GridSearchCV(estimator = pipe,
                  param_grid = grid_params,
                  scoring = 'f1',
                  cv = sss
                )
t0 = time.time()
print 'Starting timestamp: ', t0
print time.ctime()
# Fit using grid search
gs.fit(features, labels)
t1 = time.time()
print 'Ending timestamp: ', t1
print 'done in ', (t1 - t0), ' seconds'


skb_step = gs.best_estimator_.named_steps['skb']

# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in skb_step.scores_ ]

# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  skb_step.pvalues_]

# Get SelectKBest feature names, whose indices are stored in 'skb_step.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
selected_features_list = [features_list[i+1] for i in skb_step.get_support(indices=True)]
features_selected_tuple=[(features_list[i+1], feature_scores[i],
feature_scores_pvalues[i]) for i in skb_step.get_support(indices=True)]

# Sort the tuple by score, in reverse order

features_selected_tuple = sorted(features_selected_tuple, key=lambda
feature: float(feature[1]) , reverse=True)

# Print
print 'Selected Features, Scores, P-Values'
print features_selected_tuple

# Best estimator
clf = gs.best_estimator_
print 'tester classification report:'

#test_classifier(clf, my_dataset, ['poi'] + selected_features_list)
test_classifier(clf, my_dataset, features_list)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
#dump_classifier_and_data(clf, my_dataset, features_list)