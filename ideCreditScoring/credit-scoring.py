import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

SEED = 17


def impute_nan_with_median(table):
    for col in table.columns:
        table[col]= table[col].fillna(table[col].median())
    return table

def read_data ():
    data = pd.read_csv('../input/credit_scoring_sample.csv', sep=";")
    # print(data.head())

    independent_columns_names = data.columns.values
    independent_columns_names = [x for x in data if x != 'SeriousDlqin2yrs']
    print(independent_columns_names)

    table = impute_nan_with_median(data)

    X = table[independent_columns_names]
    y = table['SeriousDlqin2yrs']

    return X, y, table


def get_bootstrap_samples(data, n_samples):
    """Generate bootstrap samples using the bootstrap method."""
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples
def stat_intervals(stat, alpha):
    """Produce an interval estimate."""
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries

def bootstrap (data):
    # # Save the data about payers to split the dataset
    pay_intime = data[data['SeriousDlqin2yrs'] == 0]['MonthlyIncome'].values
    nopay_intime= data[data['SeriousDlqin2yrs'] == 1]['MonthlyIncome'].values

    # Set the seed for reproducibility of the results
    np.random.seed(SEED)

    # Generate the samples using bootstrapping and calculate the mean for each of them
    pay_intime_scores = [np.mean(sample)
                           for sample in get_bootstrap_samples(pay_intime, 2000)]
    nopay_intime_scores = [np.mean(sample)
                           for sample in get_bootstrap_samples(nopay_intime, 2000)]

    # Print the resulting interval estimates
    print("Income for good customers: mean interval",
          stat_intervals(pay_intime_scores, 0.2))
    print("Income for bad customers: mean interval",
          stat_intervals(nopay_intime_scores, 0.2))

def gradient_search (X, y):
    dt = DecisionTreeClassifier(random_state=SEED, class_weight='balanced')

    max_depth_values = [5, 6, 7, 8, 9]
    max_features_values = [4, 5, 6, 7]
    tree_params = {'max_depth': max_depth_values,
                   'max_features': max_features_values}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    gridSearch = GridSearchCV(dt, param_grid=tree_params, cv=skf, scoring='roc_auc')
    gridSearch.fit(X, y)
    print(gridSearch.best_score_)
    # print(gridSearch.cv_results_)
    return gridSearch.best_params_

def pipeline_search (X, y):
    scaler = StandardScaler()
    logit = LogisticRegression(random_state=SEED, solver='liblinear', class_weight='balanced')

    logit_pipe = Pipeline([('scaler', scaler), ('logit', logit)])
    logit_pipe_params = {'logit__C': np.logspace(-8, 8, 17)}

# def crossvaltest(model, params, train_set, train_label, n_splits=5):
#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
#
#     res = []
#     for train_index, test_index in kf.split(train_set):
#         train = train_set.iloc[train_index, :]
#         test = train_set.iloc[test_index, :]
#
#         labels = train_label.iloc[train_index]
#         test_labels = train_label.iloc[test_index]
#
#         # print(train.shape[:], labels.shape[:])
#         # print(train)
#         # print(labels)
#         # dt_clf = DecisionTreeClassifier(random_state=SEED, max_depth=3)
#         # dt_clf.fit(train, np.ravel(labels))
#
#         tree_grid = GridSearchCV(model, params)
#         tree_grid.fit(train, labels)
#
#         print(tree_grid.best_params_)
#         print(tree_grid.best_score_)
#         print(tree_grid.cv_results_)
#         # print(tree_grid.cv)
#         res.append(tree_grid.best_score_)
#         # res.append(np.mean(tuningclf.predict(test) == np.ravel(test_labels)))
#
#         test_predict = tree_grid.predict(test)
#         score = accuracy_score(test_predict, test_labels)
#         print('grid accuracy:')
#         print(score)
#     return res, score

class RandomForestClassifierCustom(BaseEstimator):
    def __init__(self, n_estimators=10, max_depth=10, max_features=10,
                 random_state=SEED):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []
        self.feat_ids_by_tree = []

    def fit(self, X, y):
        for i in range(10):
            tree = DecisionTreeClassifier(random_state=SEED+i, class_weight='balanced', max_depth=self.max_depth, max_features=self.max_features)

        pass

    def predict_proba(self, X):
        # You code here
        pass

def main ():
    X, y, df = read_data()
    print(X.head())
    print(X.shape[:], y.shape[:])

    # bootstrap(df)
    params = gradient_search(X, y)
    print(params)
    # dt = DecisionTreeClassifier(params)
    # dt.fit()


if __name__ == '__main__':
    main()
