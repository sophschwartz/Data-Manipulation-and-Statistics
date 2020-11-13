# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 10:23:37 2020

@author: schwart2
"""

# -*- coding: utf-8 -*-
"""
Independent Data Analysis of ADOS Files from NDAR.
@author: schwart2@bu.edu and Hazel
Inputs:
    ADOS .txt files directly imported from NDA.NIMH.GOV.
    # 2012 version of ADOS
    ados1_201201: ADOS-2 Mod 1
    ados2_201201: ADOS-2 Mod 2
    ados3_201201: ADOS-2 Mod 1

Outputs:
    Analysis of variables driving autism phenotypes.
"""
# %%
"""
Import packages
"""
# sklearn
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
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
from sklearn.preprocessing import LabelEncoder

# other statistics
from scipy.stats import chi2_contingency
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from kmodes import kmodes
from pandas.api.types import CategoricalDtype
from factor_analyzer import (FactorAnalyzer, ConfirmatoryFactorAnalyzer, ModelSpecificationParser)
from scipy.special import expit
import statsmodels as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif_func
from kmodes.kmodes import KModes

# Basics and plotting
import itertools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

# %%


def read_tables(file_path):
    """
    Read data file into dataframe using pandas.

    Parameters
    ----------
    file_path : path to text file

    Returns
    -------
    dataframe : pandas dataframe

    """
    df = pd.read_csv(file_path, delimiter="\t")
    return df


# %%


def extract_variables(df):
    """
    Extract variables from NDAR ADOS tables that are of interest.

    ADOS Module 1-3:
        Demographic variables: subjectkey, interview_age, sex
        Code ratings variables: Select anything with prefix 'coding',
        except remove suffix '_cmt' items (comments)
        Total Diagnostics: scoresumm_adosdiag (result from ADOS),
        scoresumm_overalldiag (clinical impression),
        scoresumm_compscore (ados css)

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
    col_diagnostic : List of strings.
        Contains column name for diagnostic code.
    """
    col_demographic = ['subjectkey', 'interview_age', 'sex']
    col_diagnostic = ['scoresumm_adosdiag', 'scoresumm_overalldiag']
    col_coding = [x for x in df.columns[df.columns.str.contains('coding')]]
    # remove non-integer valued variables with comments
    col_coding = [x for x in col_coding if not any(ignore in x for
                                                   ignore in ['cmt'])]
    col_selected = col_demographic + col_diagnostic + col_coding
    df_selected_variables = df[[c for c in df.columns if c in
                                col_selected]]
    return df_selected_variables, col_demographic, col_coding, col_diagnostic

# %%


def condense_modules(df, mod_num, col_demographics, col_diagnostic):
    """
    Condense dataframes across all modules with the same 18 codes.

    Will allow one large analysis of all data across all modules.

    Parameters
    ----------
    df : Pandas DataFrame
        includes subjectkey, interview_age, and scoresumm_adosdiag as variables

    Returns
    -------
    new_df : Pandas DataFrame with demographic codes, diagnostic labels, and
        condensed number of ADOS coding variables
    col_coding : list of strings with column names for coding variables
    col_demo_and_diag : list of strings with column names for demographic and
        diagnostic variables
    """
    # List of desired similar variables
    similar_codes_strings = ['subjectkey', 'interview_age', 'sex', 'inton',
                             'spabn', 'iecho', 'stereo', 'gest', 'ueye',
                             'faceo', 'facee', 'shrnj', 'qsov', 'asove',
                             'codingb_asov_c', 'qsres', 'imgcr', 'usens',
                             'oman', 'selfinj', 'urbeh', 'topic', 'actve',
                             'agg', 'anxty', 'scoresumm_adosdiag',
                             'scoresumm_overalldiag']

    # Lists column names in dataframe
    col_names = list(df.columns)

    # Only selects the columns from the dataframe that are in above list
    condensed_cols = [col for col in col_names if any(
        code_str in col for code_str in similar_codes_strings)]

    # Isolates columns from dataframe as new_df
    new_df = df[condensed_cols]

    # Renames the columns of new data set
    new_df.columns = ['subjectkey', 'interview_age', 'sex', 'inton', 'echo',
                      'stereo_lang', 'gest', 'eye_cont', 'facial_exp',
                      'shared_enj', 'soc_overture', 'exp_attn', 'response',
                      'imag', 'sensory', 'mannerisms', 'selfinj', 'rrb',
                      'active', 'agg', 'anx', 'scoresumm_adosdiag',
                      'scoresumm_overalldiag']

    # Identify demographics and diagnostic columns
    col_demo_and_diag = list(itertools.chain(col_demographics, col_diagnostic))

    # Coding items: not the demo or diag
    col_coding_new = [col for col in new_df.columns
                      if col not in col_demo_and_diag]

    # Adds new column to data from with module number
    new_df = new_df.assign(Module=mod_num)

    return new_df, col_coding_new

# %%


def clean_data_part1(df, col_coding):
    """
    Clean data, editing typos and removing missing data entries.

    Focus for part #1 is to focus on autism diagnosis (adosdiag)

    Cleaning includes standardizing the language used and fixing typos:
    Options for ados-2 classification should be autism spectrum
    and nonspectrum.

    Parameters
    ----------
    df: pandas dataframe
    col_coding : list of column string names containing
        independent ADOS variable codes used to predict diagnosis
    Returns
    -------
    df_cleaned: pandas dataframe
        1) Fixes typos and inconsistent entries
        2) Removes missing data
    """
    # Remove first row which is a descriptive header
    df = df.drop([0])

    # Make all items lower case
    df.scoresumm_adosdiag = df.scoresumm_adosdiag.str.lower()

    # remove invalid or missing entries
    nan_strings = ['9', 'n/a', 'low', '3', 'na']
    for string in nan_strings:
        df.loc[df['scoresumm_adosdiag'].str.contains(string),
               'scoresumm_adosdiag'] = ''
    df.scoresumm_adosdiag.replace(to_replace='', value=np.nan,
                                  inplace=True)

    # Fill NA values for Overall Diagnosis with Not Available.
    # Not necessary to remove missing data for this column.
    # df.scoresumm_overalldiag.replace(to_replace='', value='Not available',
    #                                 inplace=True)
    df.scoresumm_overalldiag.fillna('Not Available')

    # Drop remaining nan values across dataset
    df = df.dropna()

    # Replace nonspectrum alternative names
    nonspectrum_options = ['language delay', '0', 'no', 'typ', 'no diagnosis',
                           'low']
    for string in nonspectrum_options:
        df.loc[df['scoresumm_adosdiag'].str.contains(string),
               'scoresumm_adosdiag'] = 'nonspectrum'

    # Replace misspellings of spectrum
    spectrum_misspelled = ['specturm', 'spectum', 'sepctrum']
    for string in spectrum_misspelled:
        df.scoresumm_adosdiag.replace({string: 'spectrum'}, regex=True,
                                      inplace=True)

    # Replace variations of autism spectrum
    autism_spectrum_options = ['aurism', 'autsim', 'autim', 'austism',
                               'autisim', 'austim', 'asd',
                               'autism spectrum disorder', '1',
                               'autism  spectrum', 'autism-spectrum',
                               'autism spect', 'autism spectrum',
                               'autismspectrum', 'moderate',
                               'medium', 'pdd', '2', 'high',
                               'autistic disorder', 'aut']
    for string in autism_spectrum_options:
        df.loc[df['scoresumm_adosdiag'].str.contains(string),
               'scoresumm_adosdiag'] = 'autism spectrum'

    # Replace 'spectrum' and 'aut' with 'autism spectrum'; note: must match
    # exactly so as to avoid replacing nonspectrum with autism spectrum
    df.scoresumm_adosdiag.replace(to_replace='spectrum',
                                  value='autism spectrum', inplace=True)

    # Remove any participant with an age of less than 12 months. It is invalid.
    df = df[df['interview_age'].astype('int32') > 12]

    # Address errors in data: lists variables with 3 when no 3 option.
    # Replace with 2.
    df.eye_cont.replace(to_replace='3', value='2', inplace=True)
    df.gest.replace(to_replace='3', value='2', inplace=True)

    # Remove scores of 9 and 8 from dataset, considered missing data.
    mapper = {'9': np.nan, '8': np.nan}
    df.replace(mapper, inplace=True)
    df = df.dropna()

    df = clean_data_part2(df)

    return df


# %%


def clean_data_part2(df):
    """
    Focus on cleaning overall diagnosis column.

    ** NOTE: This is not very important in the grand scheme of things. It's
    just to say how many non-ASD, other neurodevelopmental disorder or
    Global Intellectual Impairment we have in sample for reporting purposes.

    Small caveat: it will pick up the occassional cases where data says
    asd + gdd but generally accurate
    """
    df.scoresumm_overalldiag = df.scoresumm_overalldiag.str.lower()
    clinical_other_options = ['dd', 'delay', 'global', 'id', 'gdd', 'adhd',
                              'anxiety']
    for string in clinical_other_options:
        df.loc[df['scoresumm_overalldiag'].str.contains(string),
               'scoresumm_overalldiag'] = 'Global Intellectual Impairment'
    return df


# %%


def format_datatypes(df, col_diagnostic, col_coding):
    """
    Format DataFrame so values are all considered categorical or ordinal.

    Relabel autism_diagnosis: 0 as autism, 1 as td (categorical int32)
    Reformat ados codes from object to ordinal int64 between 0 and 3.
    Remove 7 and 8 codes

    Parameters
    ----------
    df : Pandas DataFrame
    col_diagnostic: List of strings containing diagnostic values
    col_coding: List of strings containing column names for codes.

    Returns
    -------
    df : Pandas DataFrame
    """
    # Converts the interview age to integer type of data
    df['interview_age'] = df['interview_age'].astype('int32')
    df['sex'] = df['sex'].astype('category')

    # While 7s can be valid scores, they will not be used in this analysis.
    # Replace 7's with nan data, then removes those from data set
    mapper = {'7': np.nan}
    df = df.replace(mapper)
    df = df.dropna()

    # Assigns all code values as int64 type
    df[col_coding] = df[col_coding].apply(
        lambda x: x.astype('int64'))

    # Creates Int32 one-hot coding of data as either 0 (autism) or 1 (td)
    # for variable scoresumm_adosdiag
    lbe = LabelEncoder()
    df[col_diagnostic[0]] = lbe.fit_transform(
        df[col_diagnostic[0]])

    return df

# %%


def preprocessing_pipeline(file_path, mod_num):
    """
    Preprocess data to clean and organize.

    Parameters
    ----------
    file_name: .txt file
    mod_num: integer valued 1, 2, or 3

    Returns
    -------
    data_frameD : formatted data frame with demographics, codes, and diagnosis
    """
    data_frameA = read_tables(file_path)
    data_frameB, col_demo, col_coding, col_diag = extract_variables(
        data_frameA)
    data_frameC, col_coding_new = condense_modules(
        data_frameB, mod_num, col_demo, col_diag)
    data_frameD = clean_data_part1(data_frameC, col_coding_new)
    data_frameE = format_datatypes(data_frameD, col_diag, col_coding_new)
    return data_frameE, col_coding_new, col_demo, col_diag


def descriptive_stats(df):
    """
    Descriptive statistics about participant age and sex.

    Additional histograms showing distribution of age for each group

    Parameters
    ----------
    df : Pandas DataFrame
        Must have the following variables:
        'interview_age', 'sex', scoresumm_adosdiag', 'scoresumm_overalldiag'

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

    ax.legend(['Autism', 'Nonspectrum'])
    # df.groupby('scoresumm_adosdiag')['interview_age'].hist()
    plt.xlim((0, 300))
    plt.ylim((0, 4000))
    plt.title('Histogram of Age and Group \n (N = 10,043)')
    plt.xlabel('Age (months)')
    plt.ylabel('Frequency')

    # Identify rough estimate of participants with Global Intellectual
    # Impairment
    df_gii = df[
        df['scoresumm_overalldiag'] == 'Global Intellectual Impairment']
    print('number of GII in this dataset is', len(df_gii))

    return age_summ, sex_summ


# %%


def scree(df, col_coding_new):
    """
    Scree plot for elbow method to determine number of factors to consider.

    Parameters
    ----------
    df : Pandas DataFrame
    col_coding_new : List of strings indicating column names for coding
        variables

    Returns
    -------
    None
    """
    x_fact = df[col_coding_new]

    # Check Bartlett's test for if factor analysis should be conducted
    chi_square_value, p_value = calculate_bartlett_sphericity(x_fact)
    chi_square_value, p_value

    # Create factor analysis object and perform factor analysis
    fa = FactorAnalyzer()
    fa.fit(x_fact)
    # Check Eigenvalues
    eigen_values, vectors = fa.get_eigenvalues()
    ncomponents = len(eigen_values[eigen_values > 1])

    # Create scree plot using matplotlib
    plt.figure()
    plt.scatter(range(1, x_fact.shape[1]+1), eigen_values)
    plt.plot(range(1, x_fact.shape[1]+1), eigen_values)
    plt.xticks(np.arange(0, x_fact.shape[1], step=2))
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')

    # Assess variance of each factors and determine how much cumulative
    # variance is explained by diff number of factors
    for ncomp_range in range(1, ncomponents+1):
        fa = FactorAnalyzer()
        fa.set_params(n_factors=ncomp_range, rotation="varimax")
        print(ncomp_range)
        print(fa.fit(x_fact))
        print(fa.loadings_)
        print(fa.get_factor_variance())
        print('----------------------')
    # fa = FactorAnalyzer()
    # fa.set_params(n_factors=2, rotation="varimax")
    # print('2')
    # print(fa.fit(x_fact))
    # print(fa.loadings_)
    # print(fa.get_factor_variance())
    # print('----------------------')

    # Kaiser-Meyer-Olkin Test - suitability of data for FA (want > 0.6)
    # kmo_all,kmo_model=calculate_kmo(toAnalyze)
    # print("Kaiser-Meyer-Olkin: ")
    # print(kmo_model)

    return

# %%


def factor_analysis(df, nFactors, col_coding_new):
    """
    Exploratory multi-factor analysis using promax rotation.

    Produce table with eigenvalues and loadings
    FactorAnalyzer (https://pypi.org/project/factor-analyzer/)
    (https://www.datacamp.com/community/tutorials/introduction-#factor-analysis)
    (https://towardsdatascience.com/factor-analysis-a-complete-tutorial-1b7621890e42)

    Note: Most often, factors are rotated after extraction.  This can
    ensure that the factors are orthogonal (i.e., uncorrelated), which
    eliminates problems of multicollinearity in regression analysis.

    Eigenvalues represent the total amount of variance that can be explained
    by a given principal component. Those close to zero imply item
    collinearity.

    Parameters
    ----------
    df : Pandas DataFrame
    nFactors : Integer indicating number of factors to use for analysis
    col_coding_new : List of strings indicating column names for coding
        variables

    Returns
    -------
        efa_loadings:
            Factor loadings of all variables
        factor_comm:
            Communalities of all variables
        orig_ev:
            Eignevalies of all variables
        common_ev:
            Common eigenvalues of all variables
    """
    # Remove unneccessary columns in new df
    toAnalyze = df[col_coding_new]

    # default of FactorAnalyzer = 3, default is promax rotation
    efa = FactorAnalyzer(n_factors=nFactors)

    # Performs exploratory factor analysis
    # Defaults to using SMC
    efa.fit(toAnalyze)

    # Gives you the factor loading matrix
    # The factor loading is a matrix which shows the relationship of each variable to the underlying factor.
    # It shows the correlation coefficient for observed variable and factor.
    # It shows the variance explained by the observed variables.
    efa_loadings = efa.loadings_
    high_loadings = efa_loadings > 0.5

    model_dict = dict()
    all_high_loadings = set()
    for fac in range(0, nFactors):
        cur_loading = high_loadings[:, fac]
        cols_high_loading = list(
            itertools.compress(col_coding_new, cur_loading))
        model_dict[str(fac+1)] = cols_high_loading
        all_high_loadings.update(cols_high_loading)

    # factorLoadings.insert(0,'Factor Loadings')

    # Return array of communalities, given loadings
    # Proportion of each variable's variance explained by the factors
    # Commonalities are the sum of the squared loadings for each variable.
    # It represents the common variance.
    # It ranges from 0-1 and value close to 1 represents more variance.
    factor_comm = efa.get_communalities()
    #factorComm.insert(0,'Factor Communalities')

    # Returns both original and common eigen values
    # Eigenvalues represent variance explained each factor from the total variance.
    # It is also known as characteristic roots.
    orig_ev, common_ev = efa.get_eigenvalues()
    #orig_ev.insert(0,'Original Eigenvalues')
    #common_ev.insert(0, 'Common Eigenvalues')

    # Other possible things of interest, currently commented out
    #factorVar, propVar, cumVar = efa.get_factor_variance()
    #factorUniq = efa.get_uniquenesses()

    return efa_loadings, factor_comm, orig_ev, common_ev, model_dict, all_high_loadings

# %%


def con_factor_analysis(df, model_dict, all_high_loadings):
    """
    Confirmatory factor analysis for goodness of fit using promax rotation

    FactorAnalyzer (https://pypi.org/project/factor-analyzer/)
    (https://www.datacamp.com/community/tutorials/introduction-#factor-analysis)
    (https://towardsdatascience.com/factor-analysis-a-complete-tutorial-1b7621890e42)

    Parameters
    ----------
    df :  Pandas DataFrame
    nFactors : may not need this factor
    model_dict : dictionary with factors and high-loaded variables
    all_high_loadings : list of strings with all column names of variables
        included in factor analysis.

    Returns:
        cfa_factor_loadings: DataFrame
            The factor loadings of variables
        loadSE df:
            the standard error of factor loadings
        errorSE df:
            the standard error of error variances
        factorCOV matris:
            Covariance matrix
    """
    # Remove unneccessary columns in new df
    toAnalyze = df[list(all_high_loadings)]

    model_spec = ModelSpecificationParser.parse_model_specification_from_dict(
        toAnalyze, model_dict)

    # Performs confirmatory factor analysis
    cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
    cfa.fit(toAnalyze)

    # Gives you the factor loading matrix
    # The factor loading is a matrix which shows the relationship of each
    # variable to the underlying factor.
    # It shows the correlation coefficient for observed variable and factor.
    # It shows the variance explained by the observed variables.
    cfa_factor_loadings = cfa.loadings_

    print(cfa.factor_varcovs_)
    cfa.transform(toAnalyze.values)

    # Return array of standard errors for implied covariance
    # Move to R for RMSEA calculations for CFA performance
    # and returns standard errors for implied means
    # loadSE, errorSE = cfa.get_standard_errors()
    # factorCOV = cfa.factor_varcovs_
    # cfa_aic = cfa.aic_
    # cfa_log = cfa.log_likelihood_
    # bic = cfa.bic_

    # attempt = FactorAnalysis(n_components = 2)
    # attempt.fit(toAnalyze)
    # log = attempt.score(toAnalyze)

    return cfa_factor_loadings


# %%


def split_dataset(df, col_coding, col_diag):
    """
    Train/Test split approach for datasets above 500 rows.

    Stratification allows for the same proportion of 0, 1 y target label

    Parameters
    ----------
    df : Pandas dataframe
    col_coding : list of column string names containing
        independent variable codes used to predict diagnosis
    col_diag : column name of results of diagnoses, target?

    Returns
    -------
    x_train: pandas dataframe with int64 data types
    x_test: pandas dataframe with int64 data types
    y_train:pandas series object with int64 data types
    y_test: pandas series object with int64 data types

    """
    y = df[col_diag]
    x = df[col_coding]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2, stratify=y)
    return x_train, x_test, y_train, y_test

# %%


def logistic_regression(x_train, x_test, y_train, y_test):
    """
    Classify ASD/Non-ASD based on Features with logistic regression.

    Can only do logistic regression if make ADOS codes one-hot encoded or
    treated as continuous/ordinal data. Doing with ordinal for now.

    Note: must do this for a single factor at a time
    Load in x values specifically that are flagged as good variables
    from factor analysis.

    Run a logistic regression with all these variables and return accuracy.

    Sum the raw scores across variables to create a single x-axis predictor.
    run second regression and plot

    Using l2 regularization to force weights to be small but doesn't make
    them zero. Is particularly better for multicollinearity while keeping
    all variables.

    Parameters
    ----------
    x_train: training set of predictor codes
    x_test: test set of predictor codes
    y_train: training set of target diagnosis labels
    y_test: test set of target diagnosis labels

    Returns
    -------
    top_coeffs : coefficients over 0.5 factor loading
    odds_ratio_matrix : exponential matrix
    model : logistic regression trained model
    cm : confusion matrix

    """

    model = LogisticRegression()

    # tune hyperparameters
    c_values = [100, 10, 1.0, 0.1, 0.01]
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']

    # define grid search
    # run different modules with a k-fold cross-validation apraoch
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

    # run logistic regression based on best result from k-folds validation
    logreg = LogisticRegression(penalty=grid_result.best_params_[
                                "penalty"], C=grid_result.best_params_["C"],
                                solver=grid_result.best_params_["solver"])
    model = logreg.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # calculate accuracy, specificity, sensitivity of model here.
    # can calculate ROC later if we really want
    cm = metrics.confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:', cm)
    print('Classification Report Metrics:', metrics.classification_report(
        y_test, y_pred))
    total_1 = sum(sum(cm))
    accuracy1 = (cm[0, 0]+cm[1, 1])/total_1
    print('Logistic Regression Model Performance:')
    print('Accuracy : ', accuracy1)
    sensitivity1 = cm[0, 0]/(cm[0, 0]+cm[0, 1])
    print('Sensitivity : ', sensitivity1)
    specificity1 = cm[1, 1]/(cm[1, 0]+cm[1, 1])
    print('Specificity:', specificity1)
     # roc_val = roc(x_test, y_test, grid_result)

    # odds ratios calculated as the exponent of the model coeffecients.
    # I think this is how they might report it
    odds_ratio_matrix = np.exp(model.coef_)

    # identify features with greatest weights in regression
    code_names = pd.Index.tolist(x_train.columns)
    lr_df = pd.DataFrame(model.coef_.transpose(), code_names)
    lr_df.columns = ['Coefficient']
    top_coeffs = lr_df[abs(lr_df.Coefficient) > 0.5]

    # sum relevant code values together and plot
    # alternatives: sum together weighted fit values?
    # do some sort of decision surface transformation?
    # do some sort of svm with marginal boundary estimation?
    # reshape because it's a single feature predicting the classification
    # x_single_axis = x_train.sum(axis=1).values.reshape(-1, 1)
    # grid_result = grid_search.fit(x_single_axis, y_train)
    # logreg2 = LogisticRegression(penalty=grid_result.best_params_[
    #     "penalty"], C=grid_result.best_params_["C"],
    #     solver=grid_result.best_params_["solver"])
    # clf = logreg2.fit(x_single_axis, y_train)
    # y_pred2 = clf.predict(x_test.sum(axis=1).values.reshape(-1, 1))
    # cm = metrics.confusion_matrix(y_test, y_pred2)

    # Plot Regression Classification
    # plt.figure(1, figsize=(4, 3))
    # plt.clf()
    # plt.scatter(x_single_axis.ravel(), y_train, color='black', zorder=20)
    # x = np.linspace(0, 30, 30)
    # loss = expit(x * clf.coef_ + clf.intercept_).ravel()
    # plt.plot(x, loss, color='red', linewidth=3)
    # plt.ylabel('Diagnosis: 0 ASD 1 Non-ASD')
    # plt.xlabel('Code Total')
    # # plt.xticks(range(-5, 10))
    # plt.yticks([0, 1])
    # plt.ylim(-.25, 1.25)
    # plt.xlim(-4, 10)
    # plt.legend('Logistic Regression Model',
    # loc="upper right", fontsize='small')
    # plt.tight_layout()

    return top_coeffs, odds_ratio_matrix, model, cm


# %%

def roc(x_train, x_test, y_train, y_test, model):
    """
    Compute Receiver Operating Characteristic Curve.

    Graphical plot to illustrate diagnostic ability of a binary classifier
    system as its discrimination threshold is varied.

    Parameters
    ----------
    x_train: training set of predictor codes
    x_test: test set of predictor codes
    y_train: training set of target diagnosis labels
    y_test: test set of target diagnosis labels
    model : logistic regression trained model

    Returns
    -------
    None.

    """
    roc_auc = metrics.roc_auc_score(y_test, model.predict(x_test))
    fpr, tpr, thresholds = metrics.roc_curve(
        y_test, model.predict_proba(x_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Test Set AUC: %0.2f' % roc_auc)
    roc_auc_t = metrics.roc_auc_score(y_train, model.predict(x_train))
    fpr_t, tpr_t, thresholds = metrics.roc_curve(
        y_train, model.predict_proba(x_train)[:, 1])
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr_t, tpr_t, label='Train Set AUC: %0.2f' % roc_auc_t)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    return

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
    loadings_matrix : DataFrame with PC1 and PC2 loadings for each variable

    """
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
    accu = metrics.accuracy_score(y_test_a, y_pred)
    print('Accuracy of PCA:', accu)

    # loadings of factors
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_matrix = pd.DataFrame(
        loadings, columns=['PC1', 'PC2'], index=x_train.columns)
    # print('Largest factor loadings for PC1 \n', abs(
    #    loading_matrix['PC1']).nlargest(11))

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
    # n_samples = str(len(x_test) + len(x_train))
    ax.set_title(
        'Distinguishing Participants by Diagnosis \n in Modules 1-3 '
        '(N=10,043) at 92% Accuracy')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    for i, c, label in zip(target_ids, 'rbyrgbcmykw', target_names):
        print(i, label)
        ax.scatter(X_pca_a[y_a == i, 0], X_pca_a[y_a == i, 1],
                   c=c, label=label)
    ax.legend()

    model = LogisticRegression()

    # tune hyperparameters
    c_values = [100, 10, 1.0, 0.1, 0.01]
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']

    # define grid search
    # run different modules with a k-fold cross-validation apraoch
    grid = dict(solver=solvers, penalty=penalty, C=c_values)
    cv = model_selection.RepeatedStratifiedKFold(
        n_splits=10, n_repeats=3, random_state=1)
    grid_search = model_selection.GridSearchCV(
        estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',
        error_score=0)
    grid_result = grid_search.fit(X_train_a, y_train)

    # summarize results
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # run logistic regression based on best result from k-folds validation
    logreg = LogisticRegression(penalty=grid_result.best_params_[
                                "penalty"], C=grid_result.best_params_["C"],
                                solver=grid_result.best_params_["solver"])
    model = logreg.fit(X_train_a, y_train)
    y_pred = model.predict(X_test_a)

    # calculate accuracy, specificity, sensitivity of model here.
    # can calculate ROC later if we really want
    cm = metrics.confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:', cm)
    print('Classification Report Metrics:', metrics.classification_report(
        y_test, y_pred))
    total_1 = sum(sum(cm))
    accuracy1 = (cm[0, 0]+cm[1, 1])/total_1
    print('Logistic Regression Model Performance:')
    print('Accuracy : ', accuracy1)
    sensitivity1 = cm[0, 0]/(cm[0, 0]+cm[0, 1])
    print('Sensitivity : ', sensitivity1)
    specificity1 = cm[1, 1]/(cm[1, 0]+cm[1, 1])
    print('Specificity:', specificity1)

    return loading_matrix
# %%


def multicollinearity_check(x_train):
    """
    Check multicollinearity between variables.

    Returns
    -------
    x_train : dataframe with dummy-code independent variables (training
    data)

    Returns
    -------
    vif : Pandas DataFrame with VIF for all coding variables
    """
    x_temp = sm.tools.tools.add_constant(x_train)
    vif = pd.DataFrame()
    vif['vif factor'] = [vif_func(x_temp.values, i) for i in range(
        x_temp.values.shape[1])]
    print('multicollinearity vif factor:', vif)
    return vif
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

# %%

# Run code for the following data files
# ** Update this path for your directory (options commented below)
data_folder = Path("C://Users/schwart2/ADOSNDA/")
# data_folder = Path("Z://POLO/IndependentProjects/Sophie/ADOS/")

# Module 1
file_name = "ados1_201201.txt"
file_path = data_folder / file_name
data_frame1E, col_coding_new, col_demo, col_diag = preprocessing_pipeline(
    file_path, 1)

# Module 2
file_name = "ados2_201201.txt"
file_path = data_folder / file_name
data_frame2E, col_coding_new, col_demo, col_diag = preprocessing_pipeline(
    file_path, 2)

# Module 3
file_name = "ados3_201201.txt"
file_path = data_folder / file_name
data_frame3E, col_coding_new, col_demo, col_diag = preprocessing_pipeline(
    file_path, 3)

# All Modules, Combined
all_mods = pd.concat([data_frame1E, data_frame2E, data_frame3E])
descriptive_stats(all_mods)

scree(all_mods, col_coding_new)

# Single Factor Model
efa1_loadings, factor1_comm, orig1_ev, common1_ev, model1_dict, \
    all_high_loadings = factor_analysis(all_mods, 1, col_coding_new)
cfa1_factor_loadings = con_factor_analysis(
    all_mods, model1_dict, all_high_loadings)
loading_df = pd.DataFrame(zip(col_coding_new, efa1_loadings)).sort_values(
    by=1, axis=0, ascending=False)
loading_df.rename(columns={0: 'Code', 1: 'Loading'}, inplace=True)
print('Factor Loading : \n', loading_df)

x_train, x_test, y_train, y_test = split_dataset(
    all_mods, model1_dict['1'], col_diag[0])
vif_check = multicollinearity_check(x_train)
fm_top_coeffs, fm_odds_ratio_matrix, fm, fm_cm = logistic_regression(
    x_train, x_test, y_train, y_test)
print('Top Coefficients for 1 Factor Model are \n',
      fm_top_coeffs.sort_values(by='Coefficient'))
roc(x_train, x_test, y_train, y_test, fm)
pca_results = pca(x_train, x_test, y_train, y_test)
print('Variable Weights Contributing to PCA in 1-Factor Model \n',
      pca_results.sort_values(by='PC1', ascending=False))

# Update so does LM for all factors in 2-Factor Model

efa2_loadings, factor2_comm, orig2_ev, common2_ev, model2_dict, \
    all_high_loadings_f2 = factor_analysis(all_mods, 2, col_coding_new)
cfa2_factor_loadings = con_factor_analysis(
    all_mods, model2_dict, all_high_loadings_f2)

x_train, x_test, y_train, y_test = split_dataset(
    all_mods, list(all_high_loadings_f2), col_diag[0])
f2_top_coeffs, f2_odds_ratio_matrix, f2, f2_cm = logistic_regression(
     x_train, x_test, y_train, y_test)
print('Top Coefficients for 2-Factor Model are \n',
      f2_top_coeffs.sort_values(by='Coefficient'))
roc(x_train, x_test, y_train, y_test, f2)
pca_results_f2 = pca(x_train, x_test, y_train, y_test)
print('Variable Weights Contributing to PCA in 2-Factor Model \n',
      pca_results_f2.sort_values(by='PC1', ascending=False))

# Compare K-Means and K-Modes performance
max_score, opti_n_clusters = cluster_parameters_select(
    "k_modes", all_mods[col_coding_new])
df_cluster_kmo, df_cluster_labeled_kmo = cluster_kmodes(
    opti_n_clusters, all_mods[col_coding_new])
max_score, opti_n_clusters = cluster_parameters_select(
    "k_means", all_mods[col_coding_new])
df_cluster_kme, df_cluster_labeled_kme = cluster_kmeans(
    opti_n_clusters, all_mods[col_coding_new])

# LMs done separately for the two factors
# x_train, x_test, y_train, y_test = split_dataset(
#     all_mods, model2_dict['1'], col_diag[0])
# top_coeffs, odds_ratio_matrix, model, cm = logistic_regression(
#     x_train, x_test, y_train, y_test)
# x_train, x_test, y_train, y_test = split_dataset(
#     all_mods, model2_dict['2'], col_diag[0])
# top_coeffs, odds_ratio_matrix, model, cm = logistic_regression(
#     x_train, x_test, y_train, y_test)
