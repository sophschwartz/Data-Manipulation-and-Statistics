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
from sklearn.decomposition import FactorAnalysis
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

    Returns
    -------
    dataframe : pandas dataframe

    """
    file_to_open = data_folder / file_name
    df = pd.read_csv(file_to_open, delimiter="\t")
    return df


# %%


def extract_variables(df, version, dataset):
    """
    Extract variables from NDAR ADOS tables that are of interest.

    ADOS Module 1-3:
        Demographic variables: subjectkey, interview_age, sex
        Code ratings variables: Select anything with prefix 'coding',
        except remove suffix '_cmt' items (comments)
        Total Diagnostics: scoresumm_adosdiag, scoresumm_overalldiag,
        scoresumm_compscore

    Parameters
    ----------
    df:  pandas dataframe
        dataframe of all variables from text file.

    Returns
    -------
    df_selected_variables: Pandas DataFrame
        dataframe of all variables of interest as defined above.
    col_demographic : List of strings.
        Contains column names for demographic information.
    col_coding : List of strings.
        Contains column names for coding information.
    col_diagnostic : String.
        Contains column name for diagnostic code.
    """

    if dataset == 'ndar':
        col_demographic = ['subjectkey', 'interview_age', 'sex']

        if (version == 'toddler'):
            col_coding = [x for x in df.columns
                          [df.columns.str.contains('adost')]]
        # remove variables for comments
            col_coding_nocmt = [x for x in col_coding if not (x[-1].isalpha())]
            df.rename(columns={
                'scoresumm_rangeofconcern': 'scoresumm_adosdiag'},
                inplace=True)
            col_diagnostic = ['scoresumm_adosdiag']

        else:
            col_diagnostic = ['scoresumm_adosdiag']
            col_coding = [x for x in df.columns
                          [df.columns.str.contains('coding')]]
            # remove variables for comments
            col_coding = [x for x in col_coding if not any(ignore in x for
                                                           ignore in ['cmt'])
                          ]
    elif dataset == 'addirc':
        df = df.rename(columns={"sfari_id": "subjectkey"})
        col_demographic = ['subjectkey', 'age_months', 'a_csex']
        col_diagnostic = ['s_scoresumm_adosdiag']
        df_tmp = df.pop(
            's_scoresumm_adosdiag')  # pop and move to end
        df['s_scoresumm_adosdiag'] = df_tmp
        col_coding = [col for col in df.columns
                      [df.columns.str.contains('s_ados')]]
        col_coding = [col for col in col_coding if not any(ignore in col for
                                                           ignore in ['module',
                                                                      'algorithm'])
                      ]

    elif dataset == 'scc':
        col_demographic = ['individual', 'age_at_ados', 'sex']
        col_diagnostic = ['dx']
        col_coding = [col for col in df.columns
                      [df.columns.str.contains('coding')]]

    else:
        NameError("datatype is not defined")

    col_selected = col_demographic + col_diagnostic + col_coding
    df_selected_variables = df[[c for c in df.columns if c in
                                col_selected]]

    return df_selected_variables, col_demographic, col_coding, col_diagnostic

# %%


def clean_data(df, col_coding):
    """
    Clean data, editing typos and removing missing data entries.

    Cleaning includes standardizing the language used and fixing typos:
    Options for ados-2 classification should be autism, autism spectrum,
    and nonspectrum.

    Cleaning also includes removing invalid entries like 8s when 8 is not an
    option.

    Parameters
    ----------
    df: pandas dataframe
    col_coding : list of column string names containing
        independent ADOS variable codes used to predict diagnosis
    Returns
    -------
    df_cleaned: pandas dataframe
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
    df_for_module_analysis: pandas dataframe
        Output used for analysis of module type only.
        Does not clean df[col_coding] or require valid entries.
    """
    # Remove first row which is a descriptive header
    df = df.drop([0])

    # Make all items lower case
    df.scoresumm_adosdiag = df.scoresumm_adosdiag.str.lower()

    # Replace nonspectrum alternative names
    nonspectrum_options = ['language delay', '0', 'no', 'typ']
    for string in nonspectrum_options:
        df.loc[df['scoresumm_adosdiag'].str.contains(string),
               'scoresumm_adosdiag'] = 'nonspectrum'

    # Replace misspellings of autism
    autism_misspelled = ['aurism', 'autsim', 'autim', 'austism', 'autisim',
                         'austim']
    for string in autism_misspelled:
        df.scoresumm_adosdiag.replace({string: 'autism'}, regex=True,
                                      inplace=True)

    # Replace misspellings of spectrum
    spectrum_misspelled = ['specturm', 'spectum', 'sepctrum']
    for string in spectrum_misspelled:
        df.scoresumm_adosdiag.replace({string: 'spectrum'}, regex=True,
                                      inplace=True)

    # Replace variations of autism spectrum
    autism_spectrum_options = ['asd', 'autism spectrum disorder', '1',
                               'autism  spectrum', 'autism-spectrum',
                               'autism spect', 'autism spectrum',
                               'autismspectrum', 'moderate',
                               'medium', 'pdd']
    for string in autism_spectrum_options:
        df.loc[df['scoresumm_adosdiag'].str.contains(string),
               'scoresumm_adosdiag'] = 'autism spectrum'

    # Replace 'spectrum' with 'autism spectrum'; note: must match exactly so
    # as to avoid replacing nonspectrum with autism spectrum.
    df.scoresumm_adosdiag.replace(to_replace='spectrum',
                                  value='autism spectrum', inplace=True)

    # Replace autism alternatives with autism
    autism_options = ['aut']
    for string in autism_options:
        df.scoresumm_adosdiag.replace(to_replace=string, value='autism',
                                      inplace=True)

    autism_options = ['2', 'high', 'autistic disorder']
    for string in autism_options:
        df.loc[df['scoresumm_adosdiag'].str.contains(string),
               'scoresumm_adosdiag'] = 'autism spectrum'

    # For Toddler module: Replace typo for concern item
    df.scoresumm_adosdiag.replace(to_replace='little-to-no  concern',
                                  value='little-to-no concern',
                                  inplace=True)

    # Identify nan items in ados diagnosis column
    df.scoresumm_adosdiag.replace(to_replace='no diagnosis', value='',
                                  inplace=True)
    nan_strings = ['9', 'n/a', 'low', '3', 'na']
    for string in nan_strings:
        df.loc[df['scoresumm_adosdiag'].str.contains(string),
               'scoresumm_adosdiag'] = ''
    df.scoresumm_adosdiag.replace(to_replace='', value=np.nan,
                                  inplace=True)

    # Remove any participant with an age of less than 12 months. It is invalid.
    df = df[df['interview_age'].astype('int32') > 12]

    # Address errors in data: lists variables with 3 when no 3 option.
    # Replace with 2.
    if 'codingb_ueye_b' in df.columns:
        df.codingb_ueye_b.replace(to_replace='3', value='2', inplace=True)
    if 'codingb_ueye_a' in df.columns:
        df.codingb_ueye_a.replace(to_replace='3', value='2', inplace=True)
    if 'codinga_gest_a' in df.columns:
        df.codinga_gest_a.replace(to_replace='3', value='2', inplace=True)

    # Address errors in data: lists variables with 8 when no 8 option.
    # Replace with 9.
    # Because of all the different variables, easier just to remove all 8s
    # First: replace certain 8s with 7s to maintain them
    insufficient_codes = ['codinga_inton_a', 'codinga_iecho_a',
                          'codinga_stereo_a', 'codinga_uaothr_a',
                          'codinga_gest_a', 'codinga_spanbn_b',
                          'codinga_gest_b', 'codinga_gest_c',
                          'codingb_llnvc_c']
    for code in range(0, len(insufficient_codes)):
        if insufficient_codes[code] in df.columns:
            df[insufficient_codes[code]].replace(
                to_replace='8', value='7', inplace=True)

    #df.codingb_shrnj_b.replace(to_replace='8', value='9', inplace=True)
    #df.codingb_rname_b.replace(to_replace='8', value='9', inplace=True)

    # Only remove participants if they have missing data for age, sex, and
    # diagnosis
    df_for_module_analysis = df.dropna(
        subset=['interview_age', 'sex', 'scoresumm_adosdiag'])

    # Remove additional participants with missing data beyond diagnosis, sex,
    # and age
    mapper = {'9': np.nan, '8': np.nan}
    df.replace(mapper, inplace=True)
    df = df.dropna()

    return df, df_for_module_analysis

# %%


def relabel_autismspectrum(df, col_diagnostic):
    """
    Condense autism and autism spectrum labels into one target label.

    Replace 'autism spectrum' class label with 'autism'.

    Doing so will cause there to only be two classification target labels:
        autism or nonspectrum.

    Parameters
    ----------
    df : Pandas DataFrame.
        DataFrame must at least have a column for col_diag.
    col_diagnostic: String.
        string containing column name for diagnostic label.
        Example: ados_coldiag

    Returns
    -------
    df : Pandas DataFrame.
        DataFrame with relabeled col_diag

    """
    df[col_diagnostic].replace(to_replace='autism spectrum',
                               value='autism', inplace=True)
    return df
# %%


def format_datatypes(df, col_diagnostic):
    """
    Format DataFrame so values are all considered categorical.

    Relabel autism_diagnosis: 0 as nonspectrum and 1 as asd

    Parameters
    ----------
    df : Pandas DataFrame
        Contains the following variables:
            'interview_age', 'subjectkey', and whatever string is housed in
            the variable col_diagnostic
    col_diagnostic: String
            string containing column name for diagnostic label.
            Example: scoresumm_adosdiag


    Returns
    -------
    df : Pandas DataFrame

    """
    from sklearn.preprocessing import LabelEncoder

    df['interview_age'] = df['interview_age'].astype('int32')
    selected_cols = df.select_dtypes(include=['object']).columns
    df[selected_cols] = df[selected_cols].apply(
        lambda x: x.astype('category'))
    df['subjectkey'] = df['subjectkey'].astype('object')

    lbe = LabelEncoder()
    df[col_diagnostic] = lbe.fit_transform(
        df[col_diagnostic])

    return df


# %%

def simplify_data(df):
    """
    Take values that are categorical and make them ordinal.

    Recode so values are ordinal, between 0 and 2, and all other values are
    considered missing data.

    Doing this after format_datatypes function should be fine...

    Parameters
    ----------
    df : Pandas DataFrame
        Values include 3, 7, and 8

    Returns
    -------
    df : Pandas DataFrame
        Will no longer contain 3s, 7s, or 8 values
    """
    from pandas.api.types import CategoricalDtype
    mapper = {'7': np.nan, '8': np.nan}
    df = df.replace(mapper)
    df = df.dropna()

    mapper = {'0': 0, '1': 1, '2': 2, '3': 2}
    df = df.replace(mapper)

    return df
# %%


def condense_modules(df, mod_num):
    """
    Condense data across all modules with the same 18 codes.
    Parameters
    ----------
    df : Pandas DataFrame
        Must have subjectkey, interview_age, and scoresumm_adosdiag as variables

    Returns
    -------
    new_df : Pandas DataFrame with demographic codes, diagnostic labels, and
        condensed number of ADOS coding variables
    col_coding : list of strings with column names for coding variables
    col_demo_and_diag : list of strings with column names for demographic and
        diagnostic variables
    """
    similar_codes_strings = ['subjectkey', 'interview_age', 'sex', 'inton',
                             'spabn', 'iecho', 'stereo', 'gest', 'ueye',
                             'faceo', 'facee', 'shrnj', 'qsov', 'asove',
                             'codingb_asov_c', 'qsres', 'imgcr', 'usens',
                             'oman', 'selfinj', 'urbeh', 'topic', 'actve',
                             'agg', 'anxty', 'scoresumm_adosdiag']
    col_names = list(df.columns)
    condensed_cols = [col for col in col_names if any(
        code_str in col for code_str in similar_codes_strings)]

    new_df = df[condensed_cols]

    new_df.columns = ['subjectkey', 'interview_age', 'sex', 'inton', 'echo',
                      'stereo_lang', 'gest', 'eye_cont', 'facial_exp',
                      'shared_enj', 'soc_overture', 'exp_attn', 'response',
                      'imag', 'sensory', 'mannerisms', 'selfinj', 'rrb',
                      'active', 'agg', 'anx', 'scoresumm_adosdiag']

    col_coding = list(new_df.columns[3:-1])
    col_demo_and_diag = ['subjectkey',
                         'interview_age', 'sex', 'scoresumm_adosdiag']

    new_df = new_df.assign(Module=mod_num)

    return new_df, col_coding, col_demo_and_diag

# %%


def asd_only(df, col_diagnostic):
    """
    Select rows from dataframe in which the diagnostic label is 0 (autism)

    Parameters
    df: Pandas DataFrame
    col_diagnostic: String
            string containing column name for diagnostic label.
            Example: scoresumm_adosdiag
    ----------
    Returns
    -------
    df_asd : Pandas DataFrame
        All rows contain a col_diagnostic value of 0 (asd)

    """
    df_asd = df.loc[df[col_diagnostic] == 0]
    return df_asd
# %%


def descriptive_stats(df):
    """
    Descriptive statistics about participant age and sex.

    1. Table of number of participants with each type of ADOS
    2. Table of number of participants with each type of ADOS
    3. Identify the age at which 95% of all nonspectrum participants are a
            Mod 3, not a Mod 1 or 2.
    3. Bar plot showing distribution of severity scores for each ADOS module
    4. Table with sex and age information
    5. Return N, sex, and age information to be compared with other modules

    Parameters
    ----------
    df : Pandas DataFrame
        Must have the following variables:
            'interview_age', 'sex', scoresumm_adosdiag'.

    Returns
    -------
    age_summ: object.
        Descriptive statistics about distribution of age demographics (Months).
    sex_summ: object.
        Descriptive statistics about distribution of sex demographics
            (Male/Female).

    """
    df.info()
    sex_summ = df.groupby('scoresumm_adosdiag')['sex'].value_counts()
    age_summ = df.groupby('scoresumm_adosdiag')[
        'interview_age'].describe()

    fig, ax = plt.subplots()
    for (diagnosis, data) in df.groupby('scoresumm_adosdiag'):
        data['interview_age'].hist(alpha=0.7, ax=ax, label=diagnosis)

    ax.legend()
    # df.groupby('scoresumm_adosdiag')['interview_age'].hist()
    plt.xlim((0, 300))
    plt.ylim((0, 2000))
    plt.title('Histogram of Age and Group')
    plt.xlabel('Age (months)')
    plt.ylabel('Frequency')

    return age_summ, sex_summ

# %%


def mod3_age_distribution(df1, df2, df3):
    """
    Determine the threshold at which 90% of participants are in a module 3.

    Developed to be done with nonspectrum participants.
    P(Mod3 | Age > Threshold) = 0.9.
    P(Mod3 | Age) = P(Age | Mod3) * P(Age) / P(Mod3)
    Create a single dataframe with module (1, 2, 3) and age in months.

    Parameters
    ----------
    df1 : Data series.
        Ages for participants in module 1.
        Required variables: 'interview_age', 'module', 'scoresumm_adosdiag'
        Example: data_frame1E2
    df2 : Data series.
        Ages for participants in module 2.
            Example: data_frame2E2
    df3 : Data series.
        Ages for participants in module 3.
            Example: data_frame3E2

    Returns
    -------
    age_threshold
    """

    def binary_search(df, low_age, high_age):
        if high_age >= low_age:
            mid_age = (high_age + low_age) // 2
            age_higher = df[df['interview_age'] >= mid_age]
            if age_higher.empty:
                return binary_search(df, low_age, mid_age - 1)
            else:
                num_mod1 = len(age_higher[age_higher['module'] == 1])
                num_mod2 = len(age_higher[age_higher['module'] == 2])
                num_mod3 = len(age_higher[age_higher['module'] == 3])
                probability_mod3 = int((num_mod3 / len(age_higher))*100)
                if probability_mod3 == target:
                    return mid_age
                elif probability_mod3 > target:
                    return binary_search(df, low_age, mid_age - 1)
                else:  # if probability_mod < target
                    return binary_search(df, mid_age + 1, high_age)
        else:
            return -1

    # Nonspectrum
    nonspectrum_ages_1_fact = df1.loc[df1[
        'scoresumm_adosdiag'] == 'nonspectrum']['interview_age']
    nonspectrum_ages_2_fact = df2.loc[df3[
        'scoresumm_adosdiag'] == 'nonspectrum']['interview_age']
    nonspectrum_ages_3_fact = df3.loc[df3[
        'scoresumm_adosdiag'] == 'nonspectrum']['interview_age']
    mod1 = pd.DataFrame({'interview_age': nonspectrum_ages_1_fact, 'module': [
                        1] * len(nonspectrum_ages_1_fact)})
    mod2 = pd.DataFrame({'interview_age': nonspectrum_ages_2_fact, 'module': [
                        2] * len(nonspectrum_ages_2_fact)})
    mod3 = pd.DataFrame({'interview_age': nonspectrum_ages_3_fact, 'module': [
                        3] * len(nonspectrum_ages_3_fact)})

    df = pd.concat([mod1, mod2, mod3])
    low_age = 10
    high_age = 100
    target = 90
    nonspectrum_age_mod3_threshold = binary_search(df, low_age, high_age)

    # Spectrum
    # #take nonspectrum_age_mod3_threshold and identify proportion of asd at each module
    df1['module'] = [1] * len(df1)
    df2['module'] = [2] * len(df2)
    df3['module'] = [3] * len(df3)
    df = pd.concat([df1, df2, df3])
    asd_demo = df[df['interview_age'] >= nonspectrum_age_mod3_threshold]

    # asd_ages_1_fact = data_frame1E2.loc[data_frame1E2[
    #     'scoresumm_adosdiag'] == 'autism']['interview_age']
    # asd_ages_2_fact = data_frame2E2.loc[data_frame2E2[
    #     'scoresumm_adosdiag'] == 'autism']['interview_age']
    # asd_ages_3_fact = data_frame3E2.loc[data_frame3E2[
    #     'scoresumm_adosdiag'] == 'autism']['interview_age']
    # mod1 = pd.DataFrame({'interview_age' : asd_ages_1_fact, 'module' : [1] * len(asd_ages_1_fact)})
    # mod2 = pd.DataFrame({'interview_age' : asd_ages_2_fact, 'module' : [2] * len(asd_ages_2_fact)})
    # mod3 = pd.DataFrame({'interview_age' : asd_ages_3_fact, 'module' : [3] * len(asd_ages_3_fact)})

    # asd_n = len(pd.concat([mod1, mod2, mod3]))
    # asd_mod1_at_thresh = int((len(mod1[mod1['interview_age'] >= nonspectrum_age_mod3_threshold]) / asd_n)*100)
    # asd_mod2_at_thresh = int((len(mod2[mod2['interview_age'] >= nonspectrum_age_mod3_threshold]) / asd_n)*100)
    # asd_mod3_at_thresh = int((len(mod3[mod3['interview_age'] >= nonspectrum_age_mod3_threshold]) / asd_n)*100)

    return nonspectrum_age_mod3_threshold, asd_demo

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


def cluster_kmodes(n_clusters, df, col_coding_dummy):
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
    col_coding_dummy : list of column string names containing
        independent variable codes used to predict diagnosis

    Returns
    -------
    km.cluster_centroids_
    plot_columns

    """
    kmodes = KModes(n_clusters=n_clusters, init="Huang", n_init=2, verbose=1)
    clusters = kmodes.fit_predict(df)
    df['clusters'] = clusters
    print("Calinski-Harabasz Score",
          metrics.calinski_harabasz_score(df, clusters))
    print("Cluster Mode Centroids", kmodes.cluster_centroids_)

    # maybe send through one-hot encoding outside this function
    # df_dummy = pd.get_dummies(data_frame)
    # x = df_dummy.reset_index().values
    km = KModes(n_clusters=2, init='Huang', n_init=2, verbose=0)
    clusters = km.fit_predict(df)
    df['clusters'] = clusters
    pca = PCA(n_clusters)

    # Turn the dummified df into two columns with PCA
    plot_columns = pca.fit_transform(df)

    # Plot based on the two dimensions, and shade by cluster label
    LABEL_COLOR_MAP = {0: 'b', 1: 'r'}
    label_color = [LABEL_COLOR_MAP[l] for l in df['clusters']]
    fig1, ax1 = plt.subplots()
    plt.scatter(x=plot_columns[:, 1], y=plot_columns[:, 0], c=label_color,
                s=30)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  label='Group 1',
                                  markerfacecolor='r', markersize=15),
                       plt.Line2D([0], [0], marker='o', color='w',
                                  label='Group 2',
                                  markerfacecolor='b',
                                  markersize=15)]
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    ax1.legend(handles=legend_elements)
    print("Cluster Mode Centroids", km.cluster_centroids_)

    # check bimodal nature of clustering along first component
    fig2, ax2 = plt.subplots()
    plt.hist(plot_columns[:, 0])

    # store k-mode values
    cluster_centroids = np.roll(kmodes.cluster_centroids_, -1)
    df_cluster = pd.DataFrame(cluster_centroids,
                              columns=col_coding_dummy)

    return df_cluster, df


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


def split_dataset(data_frame, col_coding, col_diag):
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
    y = data_frame[col_diag]
    x = data_frame[col_coding]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test
# %%


# def kfold_split(data_frame, col_coding):
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
    are categorical, may or may not work with dummy variables (best with 
    continuous but we don't have')

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
    loadings_matrix

    """
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    pca = PCA(n_components=2)
    X_train_a = x_train.to_numpy()
    X_test_a = x_test.to_numpy()
    y_train_a = y_train.to_numpy().flatten()
    y_test_a = y_test.to_numpy().flatten()
    X_train_a = pca.fit_transform(X_train_a)
    X_test_a = pca.transform(X_test_a)
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train_a, y_train_a)
    y_pred = classifier.predict(X_test_a)
    cm = metrics.confusion_matrix(y_test_a, y_pred)
    print(cm)
    print('Accuracy', metrics.accuracy_score(y_test_a, y_pred))

    # loadings of factors
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_matrix = pd.DataFrame(
        loadings, columns=['PC1', 'PC2'], index=x_train.columns)
    print('Largest factor loadings for PC1', abs(
        loading_matrix['PC1']).nlargest(11))

    from matplotlib import pyplot as plt
    import matplotlib
    plt.figure(figsize=(6, 5))

    plt.rcParams.update({'font.size': 16})

    X_a = x_train.to_numpy()
    y_a = y_train.to_numpy().flatten()
    pca = PCA(n_components=2, whiten=True)
    pca.fit(X_a)
    X_pca_a = pca.transform(X_a)
    target_ids = range(0, 2)
    target_names = np.array(['Autism', 'Nonspectrum'])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(
        "Distinguishing Participants by Diagnosis \n in Modules 1-3 (N = 10,955): 93% Accuracy")
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    for i, c, label in zip(target_ids, 'rbyrgbcmykw', target_names):
        print(i, label)
        ax.scatter(X_pca_a[y_a == i, 0], X_pca_a[y_a == i, 1],
                   c=c, label=label)
    ax.legend()

    return loading_matrix


# %%


def factor_analysis(x_fact):
    """
    Linear generative model with Gaussian latent variables.

    Conduct factor analysis on x_train data. Will then conduct
    confirmatory and logistic regression on x_test...

    from sklearn.decomposition import FactorAnalysis
    fit(X[, y]): Fit the FactorAnalysis model to X using SVD based approach
    fit_transform(X[, y]): Fit to data, then transform it.
    get_covariance(): Compute data covariance with the FactorAnalysis model.
    get_params([deep]): Get parameters for this estimator.
    get_precision(): Compute data precision matrix with the FactorAnalysis 
    model. 
    score(X[, y]): Compute the average log-likelihood of the samples
    score_samples(X): Compute the log-likelihood of each sample
    set_params(**params): Set the parameters of this estimator.
    transform(X): Apply dimensionality reduction to X using the model.

    # Use Bartlett's test of sphericity to check whether variables
    intercorrelate and whether or not you should employ factor analysis. 
    If p>0.05, don't employ.

    # A scree plot shows the eigenvalues on the y-axis and the number of
    factors on the x-axis. It always displays a downward curve. The point
    where the slope of the curve is clearly leveling off (the “elbow) indicates
    the number of factors that should be generated by the analysis.

    # Also consider total amount of variability of the original variables
    explained by each factor solution - take the ones that explain the most
    (between 1 and 9 probably) - the published ADOS has 2. The original also 
    treated all variables as ordinal values, whereas I will analyze on hot
    encoded variables

    Parameters
    ----------
    data_frame : TYPE
        DESCRIPTION.
    x_train_fact
    ncomponents

    Returns
    -------
    None.

    """
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

    # Check Bartlett's test for if factor analysis should be conducted
    # cannot do with categorical variables...
    chi_square_value, p_value = calculate_bartlett_sphericity(x_fact)
    chi_square_value, p_value

    from factor_analyzer.factor_analyzer import calculate_kmo
    kmo_all, kmo_model = calculate_kmo(x_fact)
    kmo_model

    # Create factor analysis object and perform factor analysis
    # Make variables ordinal as opposed to categorical for factor analysis
    # ordered_ratings = [0, 1, 2]
    # df[col_coding] = df[col_coding].apply(lambda x: x.astype(CategoricalDtype(
    #     categories=ordered_ratings, ordered=True)))

    fa = FactorAnalyzer()
    fa.fit(x_fact)
    # Check Eigenvalues
    eigen_values, vectors = fa.get_eigenvalues()
    eigen_values
    ncomponents = len(eigen_values[eigen_values > 1])

    # Create scree plot using matplotlib
    # plt.scatter(range(1, x_fact.shape[1]+1), eigen_values)
    # plt.plot(range(1, x_fact.shape[1]+1), eigen_values)
    # plt.title('Scree Plot')
    # plt.xlabel('Factors')
    # plt.ylabel('Eigenvalue')
    # plt.grid()
    # plt.show()

    # Assess variance of each factors and determine how much cumulative
    # variance is explained by diff number of factors
    # for ncomp_range in range(1, ncomponents+1):
    #     fa = FactorAnalyzer()
    #     fa.set_params(n_factors=ncomp_range, rotation="varimax")
    #     print(ncomp_range)
    #     print(fa.fit(df))
    #     print(fa.loadings_)
    #     print(fa.get_factor_variance())
    #     print('----------------------')
    ncomponents = 2
    fa = FactorAnalyzer()
    fa.set_params(n_factors=2, rotation="varimax")
    print('2')
    print(fa.fit(x_fact))
    print(fa.loadings_)
    print(fa.get_factor_variance())
    print('----------------------')

    # transformer = FactorAnalysis(n_components=7, random_state=0)
    # X_transformed = transformer.fit_transform(X)
    # X_transformed.shape
    return fa.loadings_, ncomponents

# %%


def logistic_regression(x_train, x_test, y_train, y_test):
    """
    Can only do logistic regression if make ADOS codes binary or treat
    them as continuous.

    Gotham paper does use logistic regression to check the weighting results
    from the factor analysis to view the predictive value of scores on factors.
    "Item totals within SA and RRB factors were predictive of diagnosis" -
    report on partial log-odds coefficients

    Dimensionality reduction with PCA and LDA suggests regression would be
    done best on all variables, not reduced. Can also evaluate the same
    transform and model with different numbers of input features and choose
    the number of features (amount of dimensionality reduction) that results
    in the best average performance.

    Can do logistic regression on reduced algorithms based on factor analyses


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

    # identify features with greatest weights in regression
    code_names = pd.Index.tolist(x_train.columns)
    lr_df = pd.DataFrame(model.coef_.transpose(), code_names)
    lr_df.columns = ['Coefficient']
    top_coeffs = lr_df[abs(lr_df.Coefficient) > 1]

    return top_coeffs


# %%


def lda(x_train, x_test, y_train, y_test):
    """
    Linear Discriminant analysis on ADOS codes.

    Similar to PCA, this won't work well for categorical variables because
    assumes multivariate normal distribution of IVs.

    Determine what items are most important in distinguishing diagnostic groups
    Use sklearn.discriminant_analysis

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

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    X_train_a = x_train.to_numpy()
    X_test_a = x_test.to_numpy()
    y_train_a = y_train.to_numpy().flatten()
    y_test_a = y_test.to_numpy().flatten()

    sc = StandardScaler()
    X_train_b = sc.fit_transform(X_train_a)
    X_test_b = sc.transform(X_test_a)

    lda = LDA(n_components=1)
    X_train_c = lda.fit_transform(X_train_b, y_train_a)
    X_test_c = lda.transform(X_test_b)

    x_train_t = lda.fit_transform(x_train, y_train)
    lda_trans = lda.transform(x_test)
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(x_train_t, y_train)
    y_pred = classifier.predict(x_test_a)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy' + str(metrics.accuracy_score(y_test, y_pred)))

    # factor loadings:
    code_names = pd.Index.tolist(x_train.columns)
    ld_df = pd.DataFrame(lda.coef_.transpose(), code_names)
    ld_df.columns = ['Coefficient']
    top_coeffs = ld_df[abs(ld_df.Coefficient) > 1]
    return metrics, top_coeffs
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
    cv = model_selection.RepeatedStratifiedKFold(
        n_splits=10, n_repeats=3, random_state=1)

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
    model = SVC(kernel=grid_result.best_params_["kernel"])
    model = make_pipeline(StandardScaler(), SVC(C=grid_result.best_params_["C"],
                                                gamma=grid_result.best_params_["gamma"]))
    model.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    return


# %%
def initial_pipeline(file_name):
    """
    Documentation here
    """
    data_folder = Path("C://Users/schwart2/ADOSNDA/")
    data_frameA = read_tables(data_folder, file_name)
    data_frameB, col_demo, col_coding, col_diag = extract_variables(
        data_frameA, 'not toddler', 'ndar')
    data_frameC, df_for_module_analysis_1 = clean_data(
        data_frameB, col_coding)
    data_frameD = relabel_autismspectrum(data_frameC, col_diag)
    data_frameE = format_datatypes(data_frameD)
    return data_frameD2, data_frameE, df_for_module_analysis


def howe_validation_pipeline(df_for_module_analysis, mod_num):
    data_frameD2 = relabel_autismspectrum(df_for_module_analysis)
    data_frameE2 = format_datatypes(data_frameD2)
    data_frameE2['module'] = mod_num
    return data_frameE2


def selected_variables_pipeline(cond_codes):
    important_factors = ['inton', 'stereo_lang',
                         'rrb', 'exp_attn', 'facial_exp', 'shared_enj']
    important_factors_orig = ['gest', 'eye_cont', 'facial_exp', 'shared_enj',
                              'soc_overture', 'stereo_lang', 'sensory',
                              'mannerisms', 'rrb']
    data_frameF, col_coding_dummy = one_hot_encoding(
        cond_codes, important_factors_orig)
    x_train_fact, x_test_fact, y_train_fact, y_test_fact = split_dataset(
        data_frameF, important_factors)
    mod_coefs = logistic_regression(
        x_train_fact, x_test_fact, y_train_fact, y_test_fact)
    return mod_coefs
# %%


def ados_main():
    """
    Implementation of analysis.
    Run input file through a series of steps depending on it's module type
    Parameters
    file_name
    ----------
    Returns
    -------
    Results from analyses.
    """

    # Toddler Module
    # file_name = "adost_201201.txt"
    # data_frameTA = read_tables(data_folder, file_name)
    # data_frameTB, col_demo, col_coding, col_diag = extract_variables(
    #     data_frameTA, 'toddler', 'ndar')
    # data_frameTC = clean_data(data_frameTB, col_coding)

    # Module 1
    file_name = "ados1_201201.txt"
    data_frame1D2, data_frame1E, df_for_module_analysis_1 = initial_pipeline(
        file_name)

    # Validate Howe 2015 module distribution
    data_frame1E2 = howe_validation_pipeline(df_for_module_analysis_1, 1)

    # Validate and expand on established ADOS analyses (Gotham et al., 2007)
    # factor analysis + regression on modules separately
    data_frame1F = simplify_data(data_frame1E, col_coding)
    fa_loadings_1, ncomponents = factor_analysis(data_frame1F[col_coding])

    # Validate and expand on established ADOS analyses (Gotham et al., 2007)
    data_frame1F = simplify_data(data_frame1E, col_coding)
    age_summ_1_fact, sex_summ_1_fact = descriptive_stats(
        data_frame1F, col_demo, col_coding)

    # Doesn't do a great job. Let's simplify and condense the codes and try factor analysis again.
    cond_codes_1, col_coding = condense_modules(data_frame1F)
    fa_loadings_1, ncomponents = factor_analysis(cond_codes_1[col_coding])

    # eh, fine. Now let's one-hot encode, split the dataset and do a logistic regression using the most important variables
    mod1_coefs = selected_variables_pipeline(cond_codes_1)

    # ok now time to do it my way...
    # incl 3, 7, 8, 9 all codes
    # use one-hot encoded variables including all codes for k-modes analysis
    #cond_codes_1, col_coding, col_demo = condense_modules(data_frame1E)
    age_summ_1, sex_summ_1 = descriptive_stats(data_frame1E)
    data_frame1H, col_coding_dummy = one_hot_encoding(data_frame1E, col_coding)
    x_train_1, x_test_1, y_train_1, y_test_1 = split_dataset(
        data_frame1H, col_coding_dummy, col_diag)
    pca_results = pca(x_train_1, x_test_1, y_train_1, y_test_1)
    max_score, opti_n_clusters = cluster_parameters_select(
        "k_modes", data_frame1H[col_coding_dummy])
    df_cluster_1, df_cluster_labeled = cluster_kmodes(
        opti_n_clusters, data_frame1H[col_coding_dummy])

    # only look at asd participants?
    # autism_mask = data_frame5['diag'] == 1
    # x_train, x_test, y_train, y_test = split_dataset(data_frame5[autism_mask], col_coding)

    logistic_regression(x_train_1, x_test_1, y_train_1, y_test_1)

    # analysis on only autism

    # Module 2
    file_name = "ados2_201201.txt"
    data_frame2D2, data_frame2E, df_for_module_analysis_2 = initial_pipeline(
        file_name)

    # Validate Howe 2015 module distribution
    data_frame2E2 = howe_validation_pipeline(df_for_module_analysis_2, 2)

    # Validate and expand on established ADOS analyses (Gotham et al., 2007)
    # factor analysis + regression on modules separately
    data_frame2F = simplify_data(data_frame2E, col_coding)
    fa_loadings_2, ncomponents = factor_analysis(data_frame2F[col_coding])

    # Doesn't do a great job. Let's simplify and condense the codes and try factor analysis again.
    cond_codes_2, col_coding = condense_modules(data_frame2F)
    fa_loadings_2, ncomponents = factor_analysis(cond_codes_2[col_coding])

    # eh, fine. Now let's one-hot encode, split the dataset and do a logistic regression using the most important variables
    mod2_coefs = selected_variables_pipeline(cond_codes_2)

    # ok now time to do it my way...
    # incl 3, 7, 8, 9 all codes
    # use one-hot encoded variables including all codes for k-modes analysis
    #cond_codes_1, col_coding, col_demo = condense_modules(data_frame1E)
    age_summ_2, sex_summ_2 = descriptive_stats(data_frame2E)
    data_frame2H, col_coding_dummy = one_hot_encoding(data_frame2E, col_coding)
    x_train_2, x_test_2, y_train_2, y_test_2 = split_dataset(
        data_frame2H, col_coding_dummy, col_diag)
    pca_loadings = pca(x_train_2, x_test_2, y_train_2, y_test_2)
    max_score, opti_n_clusters = cluster_parameters_select(
        "k_modes", data_frame1H[col_coding_dummy])
    df_cluster_1, df_cluster_labeled = cluster_kmodes(
        opti_n_clusters, data_frame1H[col_coding_dummy])

    # Main analyses using one-hot encoded variables including all codes
    # (incl 3, 7, 8, 9)
    age_summ_2, sex_summ_2, data_frame2G2 = descriptive_stats(
        data_frame2E, col_demo, col_coding)
    data_frame2F, col_coding_dummy = one_hot_encoding(data_frame2E, col_coding)
    x_train, x_test, y_train, y_test = split_dataset(
        data_frame2F, col_coding_dummy)

    max_score, opti_n_clusters = cluster_parameters_select(
        "k_modes", data_frame2F[col_coding_dummy])
    df_cluster_2 = cluster_kmodes(
        opti_n_clusters, data_frame2F[col_coding_dummy])
    # only look at asd participants?
    # autism_mask = data_frame5['diag'] == 1
    # x_train, x_test, y_train, y_test = split_dataset(data_frame5[autism_mask], col_coding)

    pca_results = pca(x_train, x_test, y_train, y_test)
    svc_grid_result = svc_param_selection(x_train, y_train)
    svc_results = svc(x_train, y_train, x_test, y_test, svc_grid_result)

    # Module 3
    file_name = "ados3_201201.txt"  # 4699 usable entries
    data_frame3D2, data_frame3E, df_for_module_analysis_3 = initial_pipeline(
        file_name)

    # Validate Howe 2015 module distribution
    data_frame3E2 = howe_validation_pipeline(df_for_module_analysis_3, 3)

    # identify descriptive stats for participants above age that most NT at mod3
    select_columns = ['sex', 'interview_age', 'scoresumm_adosdiag', 'module']
    ns_mod3_thr, asd_demo = mod3_age_distribution(
        data_frame1E2[select_columns], data_frame2E2[select_columns], data_frame3E2[select_columns])
    age_summ, sex_summ, = descriptive_stats(asd_demo)
    age_summ_1_fact, sex_summ_1_fact, = descriptive_stats(
        asd_demo[asd_demo['module'] == 1])

    # Validate and expand on established ADOS analyses (Gotham et al., 2007)
    # factor analysis + regression on modules separately
    data_frame3F = simplify_data(data_frame3E, col_coding)
    fa_loadings_3, ncomponents = factor_analysis(data_frame3F[col_coding])

    # Doesn't do a great job. Let's simplify and condense the codes and try factor analysis again.
    cond_codes_3, col_coding = condense_modules(data_frame3F)
    fa_loadings_3, ncomponents = factor_analysis(cond_codes_3[col_coding])

    # eh, fine. Now let's one-hot encode, split the dataset and do a logistic regression using the most important variables
    mod3_coefs = selected_variables_pipeline(cond_codes_3)

    # ok now time to do it my way...
    # incl 3, 7, 8, 9 all codes
    # use one-hot encoded variables including all codes for k-modes analysis
    #cond_codes_1, col_coding, col_demo = condense_modules(data_frame1E)
    age_summ_3, sex_summ_3 = descriptive_stats(data_frame3E)
    data_frame3H, col_coding_dummy = one_hot_encoding(data_frame3E, col_coding)
    x_train_3, x_test_3, y_train_3, y_test_3 = split_dataset(
        data_frame3H, col_coding_dummy, col_diag)
    pca_loadings = pca(x_train_3, x_test_3, y_train_3, y_test_3)
    max_score, opti_n_clusters = cluster_parameters_select(
        "k_modes", data_frame1H[col_coding_dummy])
    df_cluster_1, df_cluster_labeled = cluster_kmodes(
        opti_n_clusters, data_frame1H[col_coding_dummy])

    # Main analyses using one-hot encoded variables including all codes
    # (incl 3, 7, 8, 9)
    age_summ_3, sex_summ_3, data_frame3G2 = descriptive_stats(
        data_frame2E, col_demo, col_coding)
    data_frame3F, col_coding_dummy = one_hot_encoding(data_frame2E, col_coding)
    x_train_3, x_test_3, y_train_3, y_test_3 = split_dataset(
        data_frame3F, col_coding_dummy)

    max_score, opti_n_clusters = cluster_parameters_select(
        "k_modes", data_frame3F[col_coding_dummy])
    cluster_centroids, cluster_columns = cluster_kmodes(
        opti_n_clusters, data_frame3F[col_coding_dummy])

    # focus on condensing data across all module tests
    ados_submain(data_frame1E, data_frame2E, data_frame3E)

    return


def ados_submain(data_frame1E, data_frame2E, data_frame3E):
    """
    Documentation here
    """

    # Condense codes and look at all samples together
    cond_codes_1, col_coding, col_demo_and_diag = condense_modules(
        data_frame1E)
    cond_codes_2, col_coding, col_demo_and_diag = condense_modules(
        data_frame2E)
    cond_codes_3, col_coding, col_demo_and_diag = condense_modules(
        data_frame3E)

    all_cond_codes = pd.concat([cond_codes_1, cond_codes_2, cond_codes_3])
    age_summ, sex_summ = descriptive_stats(all_cond_codes)
    data_frame_all, col_coding_dummy = one_hot_encoding(
        all_cond_codes, col_coding)

    x_train_all, x_test_all, y_train_all, y_test_all = split_dataset(
        data_frame_all, col_coding_dummy, col_diag)
    pca_loadings = pca(x_train_all, x_test_all, y_train_all, y_test_all)

    # unsupervised learning: k-modes and factor analysis
    # ASD only
    asd_only_all_modules = asd_only(all_cond_codes)
    asd_simplified = simplify_data(asd_only_all_modules)
    max_score, opti_n_clusters = cluster_parameters_select(
        'k_modes', asd_simplified[col_coding])
    cluster_centroids, cluster_columns = cluster_kmodes(
        4, asd_simplified[col_coding])

    #asd_only_all_df, col_coding_dummy = one_hot_encoding(asd_simplified, col_coding)
    max_score, opti_n_clusters = cluster_parameters_select(
        'k_modes', asd_only_all_df[col_coding_dummy])
    cluster_centroids, cluster_columns = cluster_kmodes(
        6, asd_only_all_df[col_coding_dummy])

    fa_loadings, n_factors = factor_analysis(asd_only_all_df[col_coding_dummy])
    max_score, opti_n_clusters = cluster_parameters_select(
        "k_modes", asd_only_all_df[col_coding_dummy])
    cluster_centroids, cluster_columns = cluster_kmodes(
        opti_n_clusters, asd_only_all_df[col_coding_dummy])
    cluster_centroids, cluster_columns = cluster_kmodes(
        6, asd_only_all_df[col_coding_dummy])

    # Supervised learning
    x_train_asd, x_test_asd, y_train_asd, y_test_asd = split_dataset(
        asd_only_all_df, col_coding_dummy, col_diag)
    pca_results = pca(x_train_all, x_test_all, y_train_all, y_test_all)
    top_coeffs = logistic_regression(
        x_train_all, x_test_all, y_train_all, y_test_all)


# %%
if __name__ == "__main__":
    ados_main()
    ados_submain()
