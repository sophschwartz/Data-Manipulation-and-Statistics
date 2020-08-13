# -*- coding: utf-8 -*-
"""
Independent Data Analysis of ADOS Files from NDAR.
@author: schwart2@bu.edu

Inputs:
    ADOS .txt files directly imported from NDA.NIMH.GOV.

Outputs:
    Analysis of variables driving autism phenotypes.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.svm import SVC
"""
NDAR-Defined file names corresponding to each form:
    #2001 version of ADOS
    ados1_200102: ADOS Mod 1
    ados2_200102: ADOS Mod 2

    # 2007 version of ADOS
    ados1_200701: ADOS (2007) Mod 1
    ados2_200701: ADOS (2007) Mod 2

    # 2012 version of ADOS
    ados1_201201: ADOS-2 Mod 1
    ados2_201201: ADOS-2 Mod 2

    same pattern as above for 3 and 4 modules

    ados_t02: ADOS Toddler Module
    adost_201201: ADOS-2 Toddler Module

    Not enough data to do Adapted ADOS
"""

# %%


def read_tables(data_folder, file_name):
    """
    Read data file into dataframe using pandas.

    Parameters
    ----------
    file_name : text file
        NDAR-derived files with ADOS variables.

    Returns
    -------
    data_frame1 : pandas dataframe
        pandas dataframe with ADOS variables

    """
    file_to_open = data_folder / file_name
    data_frame1 = pd.read_csv(file_to_open, delimiter="\t")
    return data_frame1


# %%


def extract_variables(data_frame, version):
    """
    Extract variables from ADOS tables that are of interest.

    ADOS Module 1-4:
        Demographics subjectkey, interview_age, sex
        Code ratings: Select anything with prefix 'coding',
        except remove suffix '_cmt' items (comments)
        Total Diagnostics: scoresumm_adosdiag, scoresumm_overalldiag,
        scoresumm_compscore

    Adapted ADOS Modules 1-2:
        Demographics subjectkey, interview_age, sex
        Code Ratings: ... complicated
        Total Diagnostics: Not listed?!?
    Parameters
    ----------
    data_frame:  pandas dataframe
        dataframe of all variables from text file.

    Returns
    -------
    data_frame2: pandas dataframe
        dataframe of all variables of interest as defined above.
    """
    col_demo = ['subjectkey', 'interview_age', 'sex']

    if (version == 'toddler'):
        col_coding = [x for x in data_frame.columns
                      [data_frame.columns.str.contains('adost')]]
    # remove variables for comments
        col_coding_nocmt = [x for x in col_coding if not (x[-1].isalpha())]
        data_frame.rename(columns={
            'scoresumm_rangeofconcern': 'scoresumm_adosdiag'}, inplace=True)
        col_diagnostic = ['scoresumm_adosdiag']

    else:
        col_diagnostic = ['scoresumm_adosdiag']
        col_coding = [x for x in data_frame.columns
                      [data_frame.columns.str.contains('coding')]]
        # remove variables for comments
        col_coding_nocmt = [x for x in col_coding if not any(ignore in x for
                                                             ignore in ['cmt'])
                            ]

    col_selected = col_demo + col_coding_nocmt + col_diagnostic
    data_frame = data_frame[[c for c in data_frame.columns if c in
                             col_selected]]
    # filter out any _cmt
    return data_frame, col_demo, col_coding_nocmt, col_diagnostic

# %%


def clean_data(data_frame, col_coding):
    """
    Clean data, editing typos and removing missing data entries.

    Parameters
    ----------
    data_frame: pandas dataframe
    Returns
    -------
    data_frame3: pandas dataframe
        1) removes 2nd row that is a variable description row
        2) fixes typos and inconsistent entries on the scoresumm_adosdiag to be
        one of four options: nonspectrum, autism spectrum, autism, or nan
        3) replaces 999 values with pd.nan
        4) Note: data entry error with high/mod/low indicators.
           I am taking the liberty for the following:
           it is unclear if "low" leads to an asd classification or not,
           therefore nan.
           The same is true for 'moderate' and 'high' but the likelihood is
           much higher, so marked as asd and aut respectively.
         5) Removes entries with any nan value
    """
    # first row is a descriptive header
    df = data_frame.drop([0])

    # options for ados-2 classification should be autism, autism spectrum,
    # and nonspectrum
    df.scoresumm_adosdiag = df.scoresumm_adosdiag.str.lower()

    # nonspectrum alternative names
    nonspectrum_options = ['language delay', '0', 'no', 'typ']
    for string in nonspectrum_options:
        df.loc[df['scoresumm_adosdiag'].str.contains(string),
               'scoresumm_adosdiag'] = 'nonspectrum'

    # find occurrances of mispellings and replace with the word autism
    autism_misspelled = ['aurism', 'autsim', 'autim', 'austism', 'autisim',
                         'austim']
    for string in autism_misspelled:
        df.scoresumm_adosdiag.replace({string: 'autism'}, regex=True,
                                      inplace=True)

    # spectrum misspelled
    spectrum_misspelled = ['specturm', 'spectum', 'sepctrum']
    for string in spectrum_misspelled:
        df.scoresumm_adosdiag.replace({string: 'spectrum'}, regex=True,
                                      inplace=True)

    # must just contain this item to be autism spectrum
    autism_spectrum_options = ['asd', 'autism spectrum disorder', '1',
                               'autism  spectrum', 'autism-spectrum',
                               'autism spect', 'autism spectrum',
                               'autismspectrum', 'moderate',
                               'medium', 'pdd']
    for string in autism_spectrum_options:
        df.loc[df['scoresumm_adosdiag'].str.contains(string),
               'scoresumm_adosdiag'] = 'autism spectrum'

    # matches spectrum exactly, without non before it
    df.scoresumm_adosdiag.replace(to_replace='spectrum',
                                  value='autism spectrum', inplace=True)

    # replace autism alternatives with autism
    autism_options = ['aut']
    for string in autism_options:
        df.scoresumm_adosdiag.replace(to_replace=string, value='autism',
                                      inplace=True)

    autism_options = ['2', 'high', 'autistic disorder']
    for string in autism_options:
        df.loc[df['scoresumm_adosdiag'].str.contains(string),
               'scoresumm_adosdiag'] = 'autism spectrum'

    # Concern items
    df.scoresumm_adosdiag.replace(to_replace='little-to-no  concern',
                                  value='little-to-no concern',
                                  inplace=True)

    # deal with nan items
    df.scoresumm_adosdiag.replace(to_replace='no diagnosis', value='',
                                  inplace=True)
    nan_strings = ['9', 'n/a', 'low', '3', 'na']
    for string in nan_strings:
        df.loc[df['scoresumm_adosdiag'].str.contains(string),
               'scoresumm_adosdiag'] = ''
    df.scoresumm_adosdiag.replace(to_replace='', value=np.nan,
                                  inplace=True)
    df = df.dropna()

    # replace all -9s with 9 for toddler version
    # df[col_coding] = df[col_coding].abs()
    # df2 = df.replace(to_replace=-9, value=9)

    return df


# %%


def format_datatypes(data_frame):
    """
    Format dataframe so values are all considered categorical.

    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    data_frame2.

    """
    data_frame['interview_age'] = data_frame['interview_age'].astype('int32')
    selected_cols = data_frame.select_dtypes(include=['object']).columns
    data_frame[selected_cols] = data_frame[selected_cols].apply(
        lambda x: x.astype('category'))
    data_frame['subjectkey'] = data_frame['subjectkey'].astype('object')
    return data_frame
# %%


def descriptive_stats(data_frame, col_demo):
    """
    Descriptive statistics.

    Decision is made to think of ADOS scores as categorical, not ordinal.

    1. Table of number of participants with each type of ADOS
    2. Table of number of participants with each type of ADOS
    3. Identify the age at which 95% of all nonspectrum participants are a
            Mod 3, not a Mod 1 or 2.
    3. Bar plot showing distribution of severity scores for each ADOS module
    4. Table with sex and age information
    5. Return N, sex, and age information to be compared with other modules

    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    data_frame.info()
    sex_summ = data_frame.groupby('scoresumm_adosdiag')['sex'].value_counts()
    age_summ = data_frame.groupby('scoresumm_adosdiag')[
        'interview_age'].describe()
    data_frame.groupby('scoresumm_adosdiag')['interview_age'].hist()
    plt.xlim((0, 300))
    plt.title('Histogram of Age and Group')
    plt.xlabel('Age (months)')
    plt.ylabel('Frequency')

    return age_summ, sex_summ

# %%


def one_hot_encoding(df, col_coding):
    """
    Conduct one-hot encoding of the categorical features.

    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    x_cat = df[col_coding]
    x_encoded = pd.get_dummies(x_cat)
    df2 = df.drop(col_coding, axis=1)
    df2.reset_index(drop=True, inplace=True)
    x_encoded.reset_index(drop=True, inplace=True)

    df3 = pd.concat([df2, x_encoded], axis=1)

    return df3

# %%


def split_dataset(data_frame, col_coding):
    """
    Train/Test split for datasets above 500 rows.

    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    x_train: pandas dataframe with int64 data types
    x_test: pandas dataframe with int64 data types
    y_train:pandas series object converted to a list of strings
    y_test: pandas series object converted to a list of strings

    """
    #y = data_frame['scoresumm_adosdiag'].tolist()
    y = data_frame['scoresumm_adosdiag']
    x = data_frame[col_coding]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test
# %%


def kfold_split(dataframe, col_coding):
    """
    Alternative for splitting if dataset is too small for split_dataset fx.

    Parameters
    ----------
    dataframe : TYPE
        DESCRIPTION.
    col_coding : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # evaluate model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(
        model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    kf = KFold(n_splits=10, random_state=0)
    y_hat_all = []
    for train_index, test_index in kf.split(x, y):
        reg = RandomForestRegressor(n_estimators=50, random_state=0)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = reg.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        y_hat_all.append(y_hat)
    return y_hat_all

# %%


def pca(x_train, x_test, y_train, y_test):
    """
    Principal component analysis on ADOS codes.

    Determine what items are most important in characterizing the variance.
    Normalize the data using StandardScaler, then use the PCA function
    Only do this on values that are not null.
    # **** Make sure it's doing it all with the correctly formatted items

    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    pca = PCA()
    x_train_f = pca.fit_transform(x_train)
    x_test_f = pca.transform(x_test)
    explained_variance = pca.explained_variance_ratio_
    print('Explained variance', explained_variance)

    # Plot First two components
    pca = PCA(n_components=2)
    x_train_f = pca.fit_transform(x_train)
    principal_df = pd.DataFrame(data=x_train_f, columns=['pc1', 'pc2'])
    principal_df['scoresumm_adosdiag'] = y_train
    visualize_pca(principal_df)

    # pca evaluation for 1 component
    pca = PCA(n_components=1)
    x_train_f = pca.fit_transform(x_train)
    x_test_f = pca.transform(x_test)
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier = classifier.fit(x_train_f, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test_f)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy', accuracy_score(y_test, y_pred))

    return

# %%


def visualize_pca(df_pca):
    """
    Visualize PCA with scatter plot.

    Parameters
    ----------
    df_pca : pandas dataframe
        contains first two principal components and targets.

    Returns
    -------
    None.

    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=16)
    ax.set_ylabel('Principal Component 2', fontsize=16)
    ax.set_title('2 component PCA', fontsize=20)
    targets = ['nonspectrum', 'autism spectrum', 'autism']
    colors = ['r', 'b', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = df_pca['scoresumm_adosdiag'] == target
        ax.scatter(df_pca.loc[indicesToKeep, 'pc1'],
                   df_pca.loc[indicesToKeep, 'pc2'],
                   c=color,
                   s=50)
    ax.legend(targets)
    ax.grid()
    return
# %%


def lda(x_train, x_test, y_train, y_test):
    """
    Linear Discriminant analysis on ADOS codes.

    Determine what items are most important in distinguishing diagnostic groups
    Use sklearn.discriminant_analysis

    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    lda = LinearDiscriminantAnalysis(n_components=2)
    x_train = lda.fit_transform(x_train, y_train)
    x_test = lda.transform(x_test)
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy' + str(accuracy_score(y_test, y_pred)))
    return
# %%


def svm(data_frame):
    """
    Support Vector Machine analysis on ADOS codes.

    Determine what items are most important in distinguishing diagnostic groups
    Use sklearn.svm, may or may not specify gamma and C arguments
    C: float, default 1.0. Regularzation parameter
    gamma: default = 'scale', 1
    degree: integer, default is 3

    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X, y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('svc', SVC(gamma='auto'))])
    clf = SVC(gamma=0.001, C=100.)

# %%


# def logistic_regression(x_train, x_test, y_train, y_test):
#     """
#     Logistic regression to classify groups based on ADOS code data.
#     Can't do because independent variables are not binary or continuous.

#     Dimensionality reduction with PCA and LDA suggests regression would be
#     done best on all variables, not reduced. Can also evaluate the same
#     transform and model with different numbers of input features and choose
#     the number of features (amount of dimensionality reduction) that results
#     in the best average performance.


#     Parameters
#     ----------
#     data_frame : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """

#     from sklearn.linear_model import LogisticRegression


#     # logreg = LogisticRegression()
#     # scores_logreg, report_logreg = cross_validation(x_train, y_train, logreg)
#     # print(scores_logreg)

#     logreg = LogisticRegression(C=1e5)
#     logreg.fit(x_train, y_train)
#     y_pred=logreg.predict(x_test)
#     print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
#     #print("Precision:", metrics.precision_score(y_test, y_pred))
#     #print("Recall:", metrics.recall_score(y_test, y_pred))


#     x_min, x_max = x_train[:, 0].min() - .5, x_train[:, 0].max() + .5
#     y_min, y_max = x_train[:, 1].min() - .5, x_train[:, 1].max() + .5
#     h = .02  # step size in the mesh
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.figure(1, figsize=(4, 3))
#     plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

#     # Plot also the training points
#     plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
#     plt.xlabel('Sepal length')
#     plt.ylabel('Sepal width')

#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.xticks(())
#     plt.yticks(())

#     plt.show()

#     predictions = logreg.predict(x_test)
#     score = logreg.score(x_test, y_test)
#     print(score)
#     cm = metrics.confusion_matrix(y_test, predictions)
#     print(cm)
# %%


def forest(data_frame):
    """
    Random Forest Classifier analysis on ADOS codes.

    Determine what items are most important in distinguishing diagnostic groups
    Use sklearn.ensemble.RandomForestClassifier

    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    rfc = RandomForestClassifier(n_estimators=10, max_depth=None,
                                 min_samples_split=2, random_state=0)
    scores_rfc, report_rfc = cross_validation(x, y, rfc)
    print(scores)
    model.show(view="Tree", tree_id=0)
    model.show(view="Tree", tree_id=1)

# %%


def kNeighbors(data_frame):
    """


    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
    # k = 5 for KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    scores_knn, report_knn = cross_validation(x, y, knn)
    print(scores)

# %%


def SGD(data_frame):
    """
    Logistic regression or soft-margin SVM can be done with SGDClassifier.

    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    clf = SGDClassifier(loss="log", penalty="l2", max_iter=5)

# %%

# evaluate a given model using cross-validation


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(
        model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


def cross_validation(x_train, y_train, x_test, y_test, clf):
    """
    Cross-validation for model selection.

    Use sklearn.model_selection.cross_val_score, which does splits based on
    the value of cv (here, cv=10 for 10-fold).
    Can also do sklearn.model_selection.KFold with 10 splits, but that method
    is less direct.

    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    target_names = ''

    report = classification_report(y_train, y_pred, target_names=target_names)
    scores = cross_val_score(clf, x_train, y_train, cv=10,
                             scoring='accuracy').mean()

    return scores, report

# %%


def main():
    """
    Implementation of analysis.

    Parameters
    ----------

    Returns
    -------
    Results from analyses.

    """

    data_folder = Path("C://Users/schwart2/ADOSNDA/")

    # Toddler Module
    file_name = "adost_201201.txt"
    data_frame1 = read_tables(data_folder, file_name)
    data_frame2, col_demo, col_coding, col_diag = extract_variables(
        data_frame1, 'toddler')
    data_frame3 = clean_data(data_frame2, col_coding)
    age_summ_t, sex_summ_t = descriptive_stats(data_frame3, col_coding)
    x_train, x_test, y_train, y_test = split_dataset(data_frame3, col_coding)
    pca_results = pca(x_train, x_test, y_train, y_test)
    lda_results = lda(x_train, x_test, y_train, y_test)

    # Module 1
    file_name = "ados1_201201.txt"
    data_frame1 = read_tables(data_folder, file_name)
    data_frame2, col_demo, col_coding, col_diag = extract_variables(
        data_frame1, 'not toddler')
    data_frame3 = clean_data(data_frame2, col_coding)
    age_summ_1, sex_summ_1 = descriptive_stats(data_frame3, col_demo)
    x_train, x_test, y_train, y_test = split_dataset(data_frame3, col_coding)
    pca_results = pca(x_train, x_test, y_train, y_test)
    lda_results = lda(x_train, x_test, y_train, y_test)

    # Module 2
    file_name = "ados2_201201.txt"
    data_frame1 = read_tables(data_folder, file_name)
    data_frame2, col_demo, col_coding, col_diag = extract_variables(
        data_frame1, 'not toddler')
    data_frame3 = clean_data(data_frame2, col_coding)
    data_frame4 = format_datatypes(data_frame3)
    age_summ_2, sex_summ_2 = descriptive_stats(data_frame4, col_coding)
    x_train, x_test, y_train, y_test = split_dataset(data_frame4, col_coding)
    pca_results = pca(x_train, x_test, y_train, y_test)
    lda_results = lda(x_train, x_test, y_train, y_test)

    # Module 3
    file_name = "ados3_201201.txt"  # 4699 usable entries
    data_frame1 = read_tables(data_folder, file_name)
    data_frame2, col_demo, col_coding, col_diag = extract_variables(
        data_frame1, 'not toddler')
    data_frame3 = clean_data(data_frame2, col_coding)
    age_summ_3, sex_summ_3 = descriptive_stats(data_frame3, col_coding)
    x_train, x_test, y_train, y_test = split_dataset(data_frame3, col_coding)
    pca_results = pca(x_train, x_test, y_train, y_test)
    lda_results = lda(x_train, x_test, y_train, y_test)

    return()

# get a list of models to evaluate


def get_models():
    models = dict()
    for i in range(1, 21):
        steps = [('pca', PCA(n_components=i)), ('m', LogisticRegression())]
        models[str(i)] = Pipeline(steps=steps)
    return models


# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.xticks(rotation=45)
pyplot.show()
