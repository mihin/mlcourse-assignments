import pandas as pd
import numpy as np
import pydotplus
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def tree_graph_to_png(tree, feature_names, png_file_to_save):
    tree_str = export_graphviz(tree, feature_names=feature_names,
                               filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(png_file_to_save)


def crossvaltest(model, params, train_set, train_label, n_splits=5):
    kf = KFold(n_splits=n_splits)
    res = []
    for train_index, test_index in kf.split(train_set):
        train = train_set.iloc[train_index, :]
        test = train_set.iloc[test_index, :]

        labels = train_label.iloc[train_index]
        test_labels = train_label.iloc[test_index]

        # print(train.shape[:], labels.shape[:])
        # print(train)
        # print(labels)
        # dt_clf = DecisionTreeClassifier(random_state=SEED, max_depth=3)
        # dt_clf.fit(train, np.ravel(labels))

        tree_grid = GridSearchCV(model, params)
        tree_grid.fit(train, labels)

        print(tree_grid.best_params_)
        print(tree_grid.best_score_)
        print(tree_grid.cv_results_)
        # print(tree_grid.cv)
        res.append(tree_grid.best_score_)
        # res.append(np.mean(tuningclf.predict(test) == np.ravel(test_labels)))

        test_predict = tree_grid.predict(test)
        score = accuracy_score(test_predict, test_labels)
        print('grid accuracy:')
        print(score)
    return res, score


def main():
    df = pd.read_csv('../input/mlbootcamp5_train.csv',
                     index_col='id', sep=';')
    features=[]

    df['age'] = df['age'] / 365.25
    df['age40'] = ((df['age'] >= 40) & (df['age'] < 50)).astype(int)
    df['age50'] = ((df['age'] >= 50) & (df['age'] < 55)).astype(int)
    df['age55'] = ((df['age'] >= 55) & (df['age'] < 60)).astype(int)
    df['age60'] = ((df['age'] >= 60) & (df['age'] < 65)).astype(int)
    del df['age']
    features += ['age40', 'age50', 'age55', 'age60']

    df['ap_hi120'] = ((df['ap_hi'] >= 120) & (df['ap_hi'] < 140)).astype(int)
    df['ap_hi140'] = ((df['ap_hi'] >= 140) & (df['ap_hi'] < 160)).astype(int)
    df['ap_hi160'] = ((df['ap_hi'] >= 160) & (df['ap_hi'] < 180)).astype(int)
    del df['ap_hi']
    features += ['ap_hi120', 'ap_hi140', 'ap_hi160']


    df = pd.concat([df.drop('cholesterol', axis=1),
                    pd.get_dummies(df['cholesterol'], prefix='chol')], axis=1)
    features += ['chol_1', 'chol_2', 'chol_3']

    # df = pd.concat([df.drop('gluc', axis=1), pd.get_dummies(df['gluc'], prefix='gluc')], axis=1)

    df['male'] = df['gender']-1
    del df['gender']
    features += ['male']
    features += ['smoke']

    y = df['cardio']
    X = df[features]
    # del X['cardio']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=SEED, test_size=0.7)

    print(X_train.columns)
    print(X_train.head())
    # print(y_train.head())

    dt_clf = DecisionTreeClassifier(random_state=SEED, max_depth=3)
    dt_clf.fit(X_train, y_train)

    tree_graph_to_png(tree=dt_clf, feature_names=X_train.columns,
                      png_file_to_save='./out/topic3_tree1.png')

    y_predict = dt_clf.predict(X_valid)
    origScore = accuracy_score(y_predict, y_valid)
    print('Original score:')
    print(origScore)

    # tree_params = {'max_depth': list(range(2, 11))}
    # scores, tunedScore = crossvaltest(dt_clf, tree_params, X_train, y_train, 5)
    # print('meanScore:')
    # print(np.mean(scores))
    #
    # print('score gain%:')
    # print((tunedScore - origScore) / tunedScore * 100)

    # plt.figure(figsize=(10, 8))
    # plt.scatter(range(2, 11), scores);
    # plt.plot(range(1, 12), range(0.5, 1));

    # logit_grid_searcher = GridSearchCV(param_gridestimator=logit_pipe, param_grid={'logit__C': c_values},
    #                                    scoring='roc_auc', n_jobs=4, cv=time_split, verbose=1)


SEED = 17
if __name__ == '__main__':
    main()
