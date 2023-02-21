"""
Script to perform regression analysis of the dataset.
"""
import os
import pandas as pd
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() :

    # hard-coded values
    file_questionnaire = "processed_data/FUSED_DATA_Phase3_imputation.csv"
    file_ghg_calories = "Phase 3/Phase 3 assumed calories and GHGE.csv"

    folder_results = "results"
    output_file = os.path.join(folder_results, "regression-experiments.txt")

    # load and pre-process data
    print("Loading file \"%s\"..." % file_questionnaire)
    df = pd.read_csv(file_questionnaire)
    print("Columns:", df.columns)

    print("Loading file \"%s\"..." % file_ghg_calories)
    df_ghg_cal = pd.read_csv(file_ghg_calories)
    print(df_ghg_cal)

    # if the results folder is not there, create it
    if not os.path.exists(folder_results) : os.mkdir(folder_results)

    # if the file with the results exists, delete it
    if os.path.exists(output_file) : os.remove(output_file)

    # set up and run the gridsearch
    columns_to_be_removed = ["age"]
    columns_to_be_removed.append("Q26.1_206")
    features_regression = [c for c in df.columns if c not in columns_to_be_removed]

    X = df[features_regression].values
    y = df["Q26.1_206"].values

    # this is to perform a gridsearch
    if False :
        print("Performing gridsearch...")
        from sklearn.model_selection import GridSearchCV
        param_grid = {
            'n_estimators': [10, 20, 200, 300, 500, 1000],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth' : [None, 4, 5, 6, 7, 8],
            'criterion' :['squared_error', 'absolute_error', 'poisson']
        }
        estimator = Pipeline([ ('scaler', StandardScaler()), ('randomforest', RandomForestRegressor(random_state=42)) ])
        estimator = RandomForestRegressor(random_state=42)
        CV_rfr = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=10, n_jobs=-1, verbose=3)
        CV_rfr.fit(X, y)
        print(CV_rfr.best_params_)

    # this part below is just to compare the base estimator that I used in the previous examples
    # with the alleged 'best'
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    model_best = RandomForestRegressor(n_estimators=1000, max_features='auto', max_depth=None, criterion='squared_error', n_jobs=-1, random_state=42)
    model_best_performance = []
    model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
    model_performance = []

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X) :

        print("Now working on a new fold...")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = scaler_x.fit_transform(X_train)
        X_test = scaler_x.transform(X_test)
        y_train = scaler_y.fit_transform(y_train.reshape(-1,1))
        y_test = scaler_y.transform(y_test.reshape(-1,1))

        model_best.fit(X_train, y_train)
        model.fit(X_train, y_train)

        y_pred_model_best = model_best.predict(X_test)
        y_pred_model = model.predict(X_test)

        model_best_performance.append(r2_score(y_test, y_pred_model_best))
        model_performance.append(r2_score(y_test, y_pred_model))

    import numpy as np
    print("Best model performance: %.4f +/- %.4f" % (np.mean(model_best_performance), np.std(model_best_performance)))
    print("Model performance: %.4f +/- %.4f" % (np.mean(model_performance), np.std(model_performance)))

    return

if __name__ == "__main__" :
    sys.exit( main() )
