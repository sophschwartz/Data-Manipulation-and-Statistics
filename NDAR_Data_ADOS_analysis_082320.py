# -*- coding: utf-8 -*-
"""
Independent Data Analysis of ADOS Files from NDAR.
@author: schwart2@bu.edu

Inputs:
    ADOS .txt files directly imported from NDA.NIMH.GOV.

    NDAR-Defined file names corresponding to each form:
    # 2001 version of ADOS
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

Outputs:
    Analysis of variables driving autism phenotypes.
"""


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
# from sklearn.datasets import make_blobs
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn import metrics
from sklearn.svm import SVC


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


def extract_variables(data_frame, version, dataset):
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

    if dataset == 'ndar':
        col_demo = ['subjectkey', 'interview_age', 'sex']

        if (version == 'toddler'):
            col_coding = [x for x in data_frame.columns
                          [data_frame.columns.str.contains('adost')]]
        # remove variables for comments
            col_coding_nocmt = [x for x in col_coding if not (x[-1].isalpha())]
            data_frame.rename(columns={
                'scoresumm_rangeofconcern': 'scoresumm_adosdiag'},
                inplace=True)
            col_diagnostic = ['scoresumm_adosdiag']

        else:
            col_diagnostic = ['scoresumm_adosdiag']
            col_coding = [x for x in data_frame.columns
                          [data_frame.columns.str.contains('coding')]]
            # remove variables for comments
            col_coding = [x for x in col_coding if not any(ignore in x for
                                                           ignore in ['cmt'])
                          ]
    elif dataset == 'addirc':
        data_frame = data_frame.rename(columns={"sfari_id": "subjectkey"})
        col_demo = ['subjectkey', 'age_months', 'a_csex']
        col_diagnostic = ['s_scoresumm_adosdiag']
        data_frame_tmp = data_frame.pop(
            's_scoresumm_adosdiag')  # pop and move to end
        data_frame['s_scoresumm_adosdiag'] = data_frame_tmp
        col_coding = [x for x in data_frame.columns
                      [data_frame.columns.str.contains('s_ados')]]
        col_coding = [x for x in col_coding if not any(ignore in x for
                                                       ignore in ['module',
                                                                  'algorithm'])
                      ]

    elif dataset == 'scc':
        col_demo = ['individual', 'age_at_ados', 'sex']
        col_diagnostic = ['dx']
        col_coding = [x for x in data_frame.columns
                      [data_frame.columns.str.contains('coding')]]

    else:
        NameError("datatype is not defined")

    col_selected = col_demo + col_diagnostic + col_coding
    data_frame = data_frame[[c for c in data_frame.columns if c in
                             col_selected]]
    # filter out any _cmt
    return data_frame, col_demo, col_coding, col_diagnostic

# %%


def clean_data(data_frame, col_coding):
    """
    Clean data, editing typos and removing missing data entries.

    Parameters
    ----------
    data_frame: pandas dataframe
    col_coding : list of column string names containing
        independent variable codes used to predict diagnosis
    Returns
    -------
    df: pandas dataframe
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


def relabel_autismspectrum(data_frame):
    """
    Replace 'autism spectrum' class label with 'autism'.

    Doing so will cause there to only be two classification target labels:
        autism or nonspectrum.

    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    data_frame.

    """
    data_frame.scoresumm_adosdiag.replace(to_replace='autism spectrum',
                                          value='autism', inplace=True)
    return data_frame
# %%


def format_datatypes(data_frame):
    """
    Format dataframe so values are all considered categorical.
    Relabel autism_diagnosis: 0 as nonspectrum and 1 as asd

    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    data_frame.

    """
    from sklearn.preprocessing import LabelEncoder

    data_frame['interview_age'] = data_frame['interview_age'].astype('int32')
    selected_cols = data_frame.select_dtypes(include=['object']).columns
    data_frame[selected_cols] = data_frame[selected_cols].apply(
        lambda x: x.astype('category'))
    data_frame['subjectkey'] = data_frame['subjectkey'].astype('object')

    lbe = LabelEncoder()
    data_frame["adosdiag_code"] = lbe.fit_transform(
        data_frame["scoresumm_adosdiag"])
    data_frame[["scoresumm_adosdiag", "adosdiag_code"]].head(10)

    return data_frame
# %%


def descriptive_stats(data_frame, col_demo, col_coding):
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
    col_demo : list of column string names containing
        independent variable demographic information
    col_coding : list of column string names containing
        independent variable codes used to predict diagnosis

    Returns
    -------
    age_summ: object.
    sex_summ: object.

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


def one_hot_encoding(data_frame, col_coding):
    """
    Conduct one-hot encoding of the categorical features.

    Parameters
    ----------
    data_frame : Pandas data frame
    col_coding : list of column string names containing
        independent variable codes used to predict diagnosis

    Returns
    -------
    df3: Pandas data frame

    """
    x_cat = data_frame[col_coding]
    x_encoded = pd.get_dummies(x_cat)
    df2 = data_frame.drop(col_coding, axis=1)
    df2.reset_index(drop=True, inplace=True)
    x_encoded.reset_index(drop=True, inplace=True)

    df3 = pd.concat([df2, x_encoded], axis=1)

    col_coding_dummy = x_encoded.columns

    return df3, col_coding_dummy


# %%


def cluster_kmeans(n_clusters, data_frame):
    """
    Unsupervised learning approach to identify clusters of ADOS codes.

    This model shouldn't be too computationally costly since the number of
    categorical attributes are always between 2 and 5.

    Parameters
        ----------
    n_clusters: integer value indicating the number of cluster outputs.
    data_frame: pandas dataframe housing one-hot encoded full set of data.

    Returns
    -------
    None.

    """
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(data_frame)
    print("Calinski-Harabasz Score",
          metrics.calinski_harabasz_score(data_frame, clusters))

    data_frame['clusters'] = clusters
    pca = PCA(n_clusters)
    
    # Turn the dummified df into two columns with PCA
    plot_columns = pca.fit_transform(data_frame.ix[:, 0:12])

    # Plot based on the two dimensions, and shade by cluster label
    plt.scatter(x=plot_columns[:, 1], y=plot_columns[:,
                                                     0], c=data_frame[
                                                         "clusters"], s=30)
    plt.show()
    return


def cluster_kmodes(n_clusters, data_frame):
    """
    Unsupervised learning approach to identify clusters of ADOS codes.
    # put more descriptoin here
    # look up what checks need to be put in place, how to check for spurious clusters
    # what sorts of sanity tests are possible
    # could throw in the first pca component into the analysis too
    # could do pca components as input on their own

    Parameters
        ----------
    n_clusters: integer value indicating the number of cluster outputs.
    data_frame: pandas dataframe housing one-hot encoded full set of data.
    col_coding : list of column string names containing
        independent variable codes used to predict diagnosis

    Returns
    -------
    km.cluster_centroids_
    plot_columns

    """
    kmodes = KModes(n_clusters=n_clusters, init="Huang", n_init=2, verbose=1)
    clusters = kmodes.fit_predict(data_frame)
    print("Calinski-Harabasz Score",
          metrics.calinski_harabasz_score(data_frame, clusters))
    print("Cluster Mode Centroids", kmodes.cluster_centroids_)

     # maybe send through one-hot encoding outside this function
    # df_dummy = pd.get_dummies(data_frame)
    # x = df_dummy.reset_index().values
    km = KModes(n_clusters=2, init='Huang', n_init=2, verbose=0)
    clusters = km.fit_predict(data_frame)
    data_frame['clusters'] = clusters
    pca = PCA(n_clusters)

    # Turn the dummified df into two columns with PCA
    plot_columns = pca.fit_transform(data_frame)

    # Plot based on the two dimensions, and shade by cluster label
    LABEL_COLOR_MAP = {0: 'b', 1: 'r'}
    label_color = [LABEL_COLOR_MAP[l] for l in data_frame['clusters']]
    fig1, ax1 = plt.subplots()
    plt.scatter(x=plot_columns[:, 1], y=plot_columns[:, 0], c=label_color,
                s=30)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  label='Majority Autism',
                                  markerfacecolor='r', markersize=15),
                       plt.Line2D([0], [0], marker='o', color='w',
                                  label='Majority Nonspectrum',
                                  markerfacecolor='b',
                                  markersize=15)]

    ax1.legend(handles=legend_elements)
    print("Cluster Mode Centroids", km.cluster_centroids_)

    # check bimodal nature of clustering along first component
    fig2, ax2 = plt.subplots()
    plt.hist(plot_columns[:, 0])

    # store k-mode values
    cluster_centroids = np.roll(kmodes.cluster_centroids_, -1)
    df_cluster = pd.DataFrame(cluster_centroids,
                              columns=data_frame.columns)

    return df_cluster


def cluster_parameters_select(cluster_name, data_frame):
    """
    Identify the optimal number of clusters for k-modes and means analyses.

    Parameters
        ----------
    cluster_name: string containing either 'k_means' or 'k_modes'
    data_frame: pandas dataframe housing either categorical or one-hot data

    Returns
    -------
    max_score: TYPE
    opti_n_clusters: TYPE
    """
    scores_stored = []
    if cluster_name == "k_modes":
        max_score = 0
        opti_n_clusters = 0
        cluster_num_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        # for n in range(30, 100):
        for n in cluster_num_list:
            kmodes = KModes(n_clusters=n, init="Huang", n_init=5, verbose=1)
            clusters = kmodes.fit_predict(data_frame)
            # score = metrics.calinski_harabaz_score(data, clusters)
            score = metrics.calinski_harabasz_score(data_frame, clusters)
            print("Calinski-Harabasz Score——", "n_clusters=", n, "score:",
                  score)
            if max_score < score:
                max_score = score
                opti_n_clusters = n
                scores_stored.append(score)
        print("max_score:", max_score, "opti_n_clusters:", opti_n_clusters)
        plt.plot(scores_stored)  # elbow method - select elbow point

    if cluster_name == "k_means":
        max_score = 0
        opti_n_clusters = 0
        for n in range(2, 30):
            kmodes = KModes(n_clusters=n, init="Huang", n_init=10, verbose=1)
            clusters = kmodes.fit_predict(data_frame)
            score = metrics.calinski_harabasz_score(data_frame, clusters)
            print("Calinski-Harabasz Score——", "n_clusters=", n, "score:",
                  score)
            if max_score < score:
                max_score = score
                opti_n_clusters = n
        print("max_score:", max_score, "opti_n_clusters:", opti_n_clusters)
    return max_score, opti_n_clusters

# %%


def split_dataset(data_frame, col_coding):
    """
    Train/Test split for datasets above 500 rows.

    Parameters
    ----------
    df : Pandas dataframe
    col_coding : list of column string names containing
        independent variable codes used to predict diagnosis

    Returns
    -------
    x_train: pandas dataframe with int64 data types
    x_test: pandas dataframe with int64 data types
    y_train:pandas series object with int64 data types
    y_test: pandas series object with int64 data types

    """
    y = data_frame['adosdiag_code']
    x = data_frame[col_coding]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test
# %%


#def kfold_split(data_frame, col_coding):
#    """
#    Alternative for splitting if dataset is too small for split_dataset fx.
#
#    Parameters
#    ----------
#    df : pandas dataframe
#        DESCRIPTION.
#    col_coding : list of column string names containing
#        independent variable codes used to predict diagnosis.
#
#    Returns
#    -------
#    None.
#
#    """
#    return


# %%

def pca(x_train, x_test, y_train, y_test):
    """
    Principal component analysis on ADOS codes.

    Of note, this will not work as well here because the dependent variables
    are categorical, may or may not work with dummy variables

    Determine what items are most important in characterizing the variance.
    Normalize the data using StandardScaler, then use the PCA function
    Only do this on values that are not null.

    Parameters
    ----------
    x_train : dataframe with dummy-code independent variables (training data)
    x_test : dataframe with dummy-code independent variables (testing data)
    y_train : pandas data series with dummy-code dependent variables
        (training data)
    y_test : pandas data series with dummy-code dependent variables
        (training data)

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
    visualize_pca(principal_df, x_train_f, y_train)

    # pca evaluation for 1 component
    pca = PCA(n_components=1)
    x_train_f = pca.fit_transform(x_train)
    x_test_f = pca.transform(x_test)
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier = classifier.fit(x_train_f, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test_f)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy', metrics.accuracy_score(y_test, y_pred))

    # loadings of factors
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_matrix = pd.DataFrame(
        loadings, columns=['PC1'], index=x_train.columns)
    print('Largest factor loadings', loading_matrix['PC1'].nlargest(5))

    return

# %%


def visualize_pca(df_pca, x_train_f):
    """
    Visualize PCA with scatter plot.

    Parameters
    ----------
    df_pca : pandas dataframe
        contains first two principal components and targets.
    x_train_f : scaled independent variable dataframe.

    Returns
    -------
    None.

    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=16)
    ax.set_ylabel('Principal Component 2', fontsize=16)
    ax.set_title('2 component PCA', fontsize=20)
    targets = ['nonspectrum', 'autism']
    colors = ['r', 'b', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = df_pca['scoresumm_adosdiag'] == target
        ax.scatter(df_pca.loc[indicesToKeep, 'pc1'],
                   df_pca.loc[indicesToKeep, 'pc2'],
                   c=color,
                   s=50)
    ax.legend(targets)
    ax.grid()

    score = x_train_f[:, 0:2]
    y = x_
    coeff = np.transpose(pca.components_[0:2, :])
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c=y_train)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15,
                     "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15,
                     labels[i], color='g', ha='center', va='center')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

    # histogram of first pc  - see if those numbers form peaks

    return

# %%


def logistic_regression(x_train, x_test, y_train, y_test):
    """
    Logistic regression to classify groups based on ADOS code data.
    Can't do because independent variables are not binary or continuous.

    Dimensionality reduction with PCA and LDA suggests regression would be
    done best on all variables, not reduced. Can also evaluate the same
    transform and model with different numbers of input features and choose
    the number of features (amount of dimensionality reduction) that results
    in the best average performance.


    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    model = LogisticRegression()

    # tune hyperparameters
    c_values = [100, 10, 1.0, 0.1, 0.01]
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']

    # define grid search
    grid = dict(solver=solvers, penalty=penalty, C=c_values)
    cv = model_selection.RepeatedStratifiedKFold(
        n_splits=10, n_repeats=3, random_state=1)
    grid_search = model_selection.GridSearchCV(
        estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',
        error_score=0)
    grid_result = grid_search.fit(x_train, y_train)

    # summarize results
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    # scores_logreg, report_logreg = cross_validation(x_train, y_train, logreg)
    # print(scores_logreg)

    logreg = LogisticRegression(penalty=grid_result.best_params_[
                                "penalty"], C=grid_result.best_params_["C"],
                                solver=grid_result.best_params_["solver"])
    model = logreg.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    print(metrics.classification_report(y_test, y_pred))

    return


# %%


def svc_param_selection(x_train, y_train):
    """
    Support Vector Machine analysis on ADOS codes.

    Determine what items are most important in distinguishing diagnostic groups
    Use sklearn.svm, may or may not specify gamma and C arguments for
    hyperparameter tuning (regularization, gamma)
    C: float, default 1.0. Regularzation parameter
    gamma: default = 'scale', 1
    degree: integer, default is 3

    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.

    Returns
    -------
    grid_result:

    """

    # example of grid searching key hyperparametres for SVC
    # define model and parameters
    model = SVC()
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    Cs = [50, 10, 1.0, 0.1, 0.01, 0.001]
    gammas = [0.001, 0.01, 0.1, 1]
    # gamma = ['scale']
    
    # define grid search
    grid = dict(kernel=kernels, C=Cs, gamma=gammas)
    # param_grid = {'kernel': kernels, 'C': Cs, 'gamma': gammas}
    
    # change cv?
    cv = model_selection.RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Grid search    
    grid_search = model_selection.GridSearchCV(estimator=SVC, param_grid=grid,
                               n_jobs=-1, cv=cv, scoring='accuracy',
                               error_score=0)
    grid_result = grid_search.fit(x_train, y_train)
    
    # summarize results
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result


def svc(x_train, y_train, x_test, x_train, grid_result):
    """
    Create SVM Classifier
    """
    grid_result.best_params_
    model = SVC(kernel = grid_result.best_params_["kernel"])
    model = make_pipeline(StandardScaler(), SVC(C=grid_result.best_params_["C"],
                                gamma=grid_result.best_params_["gamma"]))
    model.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    return



# %%


def ados_main():
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
    # file_name = "adost_201201.txt"
    # data_frame1 = read_tables(data_folder, file_name)
    # data_frame2, col_demo, col_coding, col_diag = extract_variables(
    #     data_frame1, 'toddler')

    # # Module 1
    # file_name = "ados1_201201.txt"

    # Module 2
    file_name = "ados2_201201.txt"
    data_frame1A = read_tables(data_folder, file_name)
    data_frame2A, col_demo, col_coding, col_diag = extract_variables(
        data_frame1A, 'not toddler', 'ndar')
    # file_name = "addirc_Dataset_2012_ados_2.txt"
    # data_frame1B = read_tables(data_folder, file_name)
    # data_frame2B, col_demo, col_coding, col_diag = extract_variables(
    #     data_frame1B, 'not toddler', 'addirc')
    # new_cols = {x: y for x, y in zip(data_frame2B.columns, data_frame2A.columns)}
    # data_frame2 = data_frame2A.append(data_frame2B.rename(columns = new_cols))
    data_frame3 = clean_data(data_frame2A, col_coding)
    data_frame4 = relabel_autismspectrum(data_frame3)
    data_frame5 = format_datatypes(data_frame4)
    data_frame6, col_coding_dummy = one_hot_encoding(data_frame5, col_coding)
    age_summ_2, sex_summ_2 = descriptive_stats(
        data_frame5, col_demo, col_coding)
    max_score, opti_n_clusters = cluster_parameters_select("k_modes", data_frame6[col_coding])
    cluster_centroids, cluster_columns = cluster_kmodes(opti_n_clusters, data_frame6[col_coding])
    x_train, x_test, y_train, y_test = split_dataset(data_frame6, col_coding_dummy)
    # only look at asd participants?
    # autism_mask = data_frame5['diag'] == 1
    # x_train, x_test, y_train, y_test = split_dataset(data_frame5[autism_mask], col_coding)
    
    pca_results = pca(x_train, x_test, y_train, y_test)
    svc_grid_result = svc_param_selection(x_train, y_train)
    svc_results = svc(x_train, y_train, x_test, y_test, svc_grid_result)


#    # Module 3
#    file_name = "ados3_201201.txt"  # 4699 usable entries



# %%
if __name__ == "__main__":
    ados_main()