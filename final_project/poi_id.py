'''
Report: KNeighborsClassifier

done in  1937.33286881  seconds
Selected Features, Scores, P-Values
[('exercised_stock_options', '27.45', '0.000'), ('total_stock_value', '23.67', '0.000'), ('bonus', '15.80', '0.000'), ('salary', '10.90', '0.001'), ('deferred_income', '10.29', '0.002'), ('restricted_stock', '8.46', '0.004'), ('total_payments', '8.41', '0.004'), ('long_term_incentive', '8.36', '0.004'), ('shared_receipt_with_poi', '7.48', '0.007'), ('from_poi_to_this_person', '4.28', '0.040'), ('other', '3.96', '0.049'), ('poi_messages_total_messages_ratio', '3.87', '0.051'), ('loan_advances', '3.85', '0.052')]
tester classification report:
['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'poi_messages_total_messages_ratio']
/Applications/anaconda/lib/python2.7/site-packages/sklearn/feature_selection/univariate_selection.py:113: UserWarning: Features [3] are constant.
  UserWarning)
/Applications/anaconda/lib/python2.7/site-packages/sklearn/feature_selection/univariate_selection.py:114: RuntimeWarning: invalid value encountered in divide
  f = msb / msw
Pipeline(memory=None,
     steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=False)), ('skb', SelectKBest(k=13, score_func=<function f_classif at 0x1170b6848>)), ('pca', PCA(copy=True, iterated_power='auto', n_components=1, random_state=42,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='distance'))])
  Accuracy: 0.87080 Precision: 0.52120  Recall: 0.38100 F1: 0.44021 F2: 0.40266
  Total predictions: 15000  True positives:  762  False positives:  700 False negatives: 1238 True negatives: 12300


Report 100-fold cross-validation when training, 1000-fold cv when testing
Ending timestamp:  1531407364.26
done in  9018.51136398  seconds
Best score from the pipeline:  0.379857142857
Feature Ranking:
feature no. 1: poi_messages_total_messages_ratio (0.152698924925)
feature no. 2: total_payments (0.12825665974)
feature no. 3: total_stock_value (0.127183785477)
feature no. 4: long_term_incentive (0.10263509473)
feature no. 5: exercised_stock_options (0.0844120640939)
feature no. 6: deferred_income (0.0731560200504)
feature no. 7: restricted_stock (0.0729123903489)
feature no. 8: bonus (0.0588591716643)
feature no. 9: expenses (0.0402617974295)
feature no. 10: from_poi_to_this_person (0.0374883652549)
feature no. 11: salary (0.0291464081023)
feature no. 12: to_messages (0.0268306710977)
feature no. 13: other (0.0202748518273)
feature no. 14: deferral_payments (0.0138725747541)
feature no. 15: from_messages (0.0101448131742)
feature no. 16: director_fees (0.00998833683408)
feature no. 17: shared_receipt_with_poi (0.00620816501592)
feature no. 18: from_this_person_to_poi (0.00329180032199)
feature no. 19: restricted_stock_deferred (0.00237810515773)
feature no. 20: loan_advances (0.0)
tester classification report:
['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'poi_messages_total_messages_ratio']
Pipeline(memory=None,
     steps=[('clf', ExtraTreesClassifier(bootstrap=False, class_weight='balanced',
           criterion='gini', max_depth=None, max_features='sqrt',
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=3,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=10, n_jobs=-1, oob_score=False, random_state=42,
           verbose=0, warm_start=False))])
  Accuracy: 0.80167 Precision: 0.30979  Recall: 0.39700 F1: 0.34802 F2: 0.37584
  Total predictions: 15000  True positives:  794  False positives: 1769 False negatives: 1206 True negatives: 11231

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
    df[feature] = df[feature].fillna(df[feature].median())


# # Replacing NaNs in email features with 0
for feature in email_features:
    df[feature] = df[feature].astype(float)
    df[feature] = df[feature].fillna(df[feature].median())

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Task 2: Remove outliers
### Task 3: Create new feature poi_messages_total_messages_ratio.

df = df.drop('TOTAL')
df = df.drop('THE TRAVEL AGENCY IN THE PARK')
print 'outliers dropped.'

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
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size = 0.3, random_state = 42, stratify = labels)



#### ExtraTreesClassifier ####
'''
pipe = Pipeline(
                [
                 ('clf', ExtraTreesClassifier())
                ]
               )
sss = StratifiedShuffleSplit(n_splits = 100, random_state = 42)


##testing grid_params
grid_params = [
                {'clf__n_estimators': [10, 20, 30, 40, 50],
                 'clf__criterion': ['gini', 'entropy'],
                 'clf__max_features': ['sqrt'],
                 'clf__max_depth': [None],
                 'clf__min_samples_split': [2, 3, 4],
                 'clf__min_samples_leaf': [1, 2, 3],
                 'clf__min_weight_fraction_leaf': [0.0],
                 'clf__max_leaf_nodes': [None],
                 'clf__bootstrap': [False],
                 'clf__oob_score': [False],
                 'clf__n_jobs': [-1],
                 'clf__random_state': [42],
                 'clf__class_weight':['balanced', 'balanced_subsample']
                }
              ]

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
# Since the size of the data is fairly small, here I used the full dataset with cross validation
gs.fit(features, labels)
t1 = time.time()
print 'Ending timestamp: ', t1
print 'done in ', (t1 - t0), ' seconds'
print 'Best score from the pipeline: ', gs.best_score_
clf = gs.best_estimator_
importances = clf.named_steps['clf'].feature_importances_
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(0, len(features_list) - 1):
    print "feature no. {}: {} ({})".format(i+1,features_list[indices[i]+1],importances[indices[i]])

'''
############### End ExtraTreesClassifier ############################################################

#### KNeighborsClassifier ####
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
### Task 6: Dump your classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, features_list)



