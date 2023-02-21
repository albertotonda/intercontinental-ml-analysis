"""
Script to perform an analysis of the dataset.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
#import seaborn as sns
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold

#sns.set_theme() # prettier graphics

def cross_validation_classification(X, y, n_splits, random_state, figure_path="roc.png") :

    performance = {"accuracy_train": [], "accuracy_test": [], "f1_train": [], "f1_test": []}
    most_relevant_features = []
    y_true_values = []
    y_pred_values = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = [ [fold_index, train_index, test_index] for fold_index, [train_index, test_index] in enumerate(skf.split(X, y)) ]

    # this stuff below is later used to plot the ROC curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # let's immediately create the figure
    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)

    for fold_index, train_index, test_index in folds :
        print("Now working on fold #%d, preparing data..." % fold_index)

        classifier = RandomForestClassifier(n_estimators=300, random_state=random_state)
        # 'best parameters' found through grid search
        #classifier = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, random_state=random_state)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # train and test
        print("Training classifier...")
        classifier.fit(X_train, y_train)

        print("Testing classifier...")
        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)

        f1_train = f1_score(y_train, y_train_pred)
        f1_test = f1_score(y_test, y_test_pred)
        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)

        # store performance information
        performance["f1_train"].append(f1_train)
        performance["f1_test"].append(f1_test)
        performance["accuracy_train"].append(accuracy_train)
        performance["accuracy_test"].append(accuracy_test)
        y_true_values.extend(y_test)
        y_pred_values.extend(y_test_pred)

        # some output
        print("Train F1 score: %.4f ; Test F1 score: %.4f" % (f1_train, f1_test))
        print("Train accuracy: %.4f ; Test accuracy: %.4f" % (accuracy_train, accuracy_test))

        # also store the most relevant features
        most_relevant_features.append(classifier.feature_importances_)

        # stuff that is used to plot the figure
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X_test,
            y_test,
            name="ROC fold %d" % fold_index,
            alpha=0.3,
            lw=1,
            ax=ax,
        )

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    # wrap up the figure and save it
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="ROC for a 10-fold cross-validation",
    )
    ax.legend(loc="lower right")

    plt.savefig(figure_path, dpi=300)
    plt.close(fig)

    return performance, y_true_values, y_pred_values, most_relevant_features

def save_feature_importance(feature_names, most_relevant_features, results_folder, file_name) :

    df_dict = dict()
    df_dict["feature_name"] = feature_names
    for i in range(0, len(most_relevant_features)) : df_dict["fold_%d" % i] = most_relevant_features[i]

    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(os.path.join(results_folder, file_name), index=False)

    return

def main() :

    # hard-coded values
    file_questionnaire = "processed_data/FUSED_DATA_Phase3_imputation.csv"
    file_income = "income_data/simplified_income_data.csv"
    file_ghg_calories = "Phase 3/Phase 3 assumed calories and GHGE.xlsx"

    folder_results = "results"

    question_income_individual = "Q2.9_1"
    question_income_household = "Q2.9_2"
    question_country = "Q2.1"

    # scikit-learn stuff
    n_estimators = 300
    n_splits = 10
    random_state = 42

    # dictionary with all country codes
    countries = dict()
    countries[7] = "Argentina"
    countries[24] = "Brazil"
    countries[37] = "Colombia"
    countries[66] = "Ghana"
    countries[78] = "India"
    countries[135] = "Peru"
    countries[185] = "United Kingdom"

    # load and pre-process data
    print("Loading file \"%s\"..." % file_questionnaire)
    df = pd.read_csv(file_questionnaire)

    # if the results folder is not there, create it
    if not os.path.exists(folder_results) : os.mkdir(folder_results)

    # this part will be moved, for the moment it's just an analysis of income
    questions_income = {"individual" : question_income_individual, "household" : question_income_household}
    for q in questions_income :
        for c in countries :
            print("Now analyzing income data for %s (%d), question \"%s\"..." % (countries[c], c, questions_income[q]))

            df_c = df[df[question_country] == c]
            print(df_c[questions_income[q]])
            
            fig, ax = plt.subplots()
            ax.set_title("Income distribution in the dataset for %s" % countries[c])
            n, bins, patches = ax.hist(df_c[questions_income[q]].values, bins=20)
            plt.savefig(os.path.join(folder_results, "income-%s-%s.png" % (q, countries[c])), dpi=300)
            #plt.show()
            plt.close(fig)

    # now, we start from the classification
    print("Starting from classification run")

    threshold = 5
    people_over_threshold = 0
    people_did_not_reply = 0

    # questions of interest
    questions = [x for x in list(df) if x.startswith("Q24")]

    # rows to be removed, they did not answer to the questions of interest
    remove_row = []
    for index, row in df.iterrows() :
        person_over_threshold = False
        not_answered = 0
        for q in questions :
            if row[q] != "-1" and int(row[q]) >= threshold :
                person_over_threshold = True
            elif row[q] == "-1" :
                not_answered += 1
        
        if person_over_threshold : 
            people_over_threshold += 1
        
        if not_answered == len(questions) : 
            people_did_not_reply += 1
            remove_row.append(index)

    print("I found %d/%d people that gave an answer over the threshold of %d to at least one of the following questions: \"%s\""
      % (people_over_threshold, len(df), threshold, str(questions)))
    print("%d people did not reply to any of the questions." % people_did_not_reply)

    print("Filtering dataframe from people who did not answer, and preparing classification problem...")
    df.drop(remove_row, inplace=True)
    print("Dataset now contains %d rows." % df.shape[0])

    # prepare labels
    y = np.zeros(len(df))
    for index, row in df.iterrows() :
        for q in questions :
            if int(row[q]) >= threshold :
                y[index] = 1

    print("With threshold %d, found %d people below threshold (class 0), and %d people above (class 1)" %
      (threshold, len(y)-np.count_nonzero(y), np.count_nonzero(y)))

    print("Removing columns related to Q24 and 'age'...")
    features_classification = [x for x in df.columns if x not in questions and x != 'age']
    df_classification = df[features_classification]
    print("With threshold >=%d, the final dataset used for classification will have %d rows (samples) and %d columns (features)" %
      (threshold, df_classification.shape[0], df_classification.shape[1]))

    # features
    X = df_classification.values

    # temporary thing here, let's try a gridsearch!
    perform_grid_search = False
    if perform_grid_search == True :
        print("Performing gridsearch...")
        from sklearn.model_selection import GridSearchCV
        param_grid = {
            'n_estimators': [10, 20, 200, 300, 500, 1000],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth' : [None, 4, 5, 6, 7, 8],
            'criterion' :['gini', 'entropy']
        }
        CV_rfc = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=10, n_jobs=-1, verbose=1)
        CV_rfc.fit(X, y)
        print(CV_rfc.best_params_)

    # run the cross-validation
    performance, y_true_values, y_pred_values, most_relevant_features = cross_validation_classification(X, y, n_splits, random_state, figure_path="classification-global-ROC.png")
    for metric in performance :
        print("Mean %s: %.4f +/- %.4f" % (metric, np.mean(performance[metric]), np.std(performance[metric])))

    # save the results
    save_feature_importance(features_classification, most_relevant_features, folder_results, "classification-feature-importance.csv")

    # what happens if we do the same thing, but removing one country at a time?
    for c in countries :
        print("Now checking what happens if we remove %s from the dataset" % countries[c])

        df_selected = df[df[question_country] != c]
        df_selected.reset_index(inplace=True)

        X = df_selected[features_classification].values
        y = np.zeros(len(df_selected))
        for index, row in df_selected.iterrows() :
            for q in questions :
                if int(row[q]) >= threshold :
                    y[index] = 1

        performance, y_true_values, y_pred_values, most_relevant_features = cross_validation_classification(X, y, n_splits, random_state)
        for metric in performance :
            print("Mean %s: %.4f +/- %.4f" % (metric, np.mean(performance[metric]), np.std(performance[metric])))

        save_feature_importance(features_classification, most_relevant_features, folder_results, "classification-no-%s-feature-importance.csv" % countries[c])

    return

if __name__ == "__main__" :
    sys.exit( main() )
