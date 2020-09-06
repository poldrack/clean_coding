# Refactored version of example1.py
# load survey data, perform factor analysis, and compare to mental health data


import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import scale
from collections import OrderedDict, namedtuple


# load health data
def load_health_data(datadir, filename='health.csv'):
    return(pd.read_csv(datadir / filename, index_col=0))


# remove unwanted variables from health data, keeping only mental health scales
def select_wanted_health_variables(health_data_raw, vars_to_use=None):
    if vars_to_use is None:
        vars_to_use = ('Nervous', 'Hopeless',
                       'RestlessFidgety', 'Depressed',
                       'EverythingIsEffort', 'Worthless')
    return(health_data_raw[list(vars_to_use)])


# load behavioral data (which includes survey and task data)
def load_behavioral_data(datadir, filename='meaningful_variables_clean.csv'):
    return(pd.read_csv(datadir / filename, index_col=0))


# extract survey data from behavioral data
# survey variables are labeled <survey_name>_survey.<variable name>
# so filter on variables that include "_survey" in their name
def extract_surveys_from_behavioral_data(behavioral_data_raw):
    survey_variables = [i for i in behavioral_data_raw.columns if i.find('_survey') > -1]
    return(behavioral_data_raw[survey_variables])


# align datasets, ensuring that order of subjects is the same
# subject code is stored in the data frame index
# NB: in this case we know they are aligned, so I am just going to add an
# assertion to make sure that they are aligned
# if I run into a case where it fails then I would write another function
# to align them - but not until it's needed!
def confirm_data_frame_index_alignment(df1, df2):
    assert all(df1.index == df2.index)


# remove subjects with missing data for either set of variables
def remove_NA_rows_from_matched_data_frames(df1, df2):
    indices_to_retain = list(set(df1.dropna().index).intersection(df2.dropna().index))
    return(df1.loc[indices_to_retain, :], df2.loc[indices_to_retain, :])


# perform FA across a range of n_components and choose dimensionality with mimimum AIC
def get_best_FA_dimensionality(data, maxdims=None):
    if maxdims is None:
        maxdims = data.shape[1]

    AIC_by_components = OrderedDict()

    for n_components in range(1, maxdims + 1):
        fa = fit_and_score_factor_analysis(data, n_components)
        AIC_by_components[n_components] = fa.AIC

    minimum_AIC = min(AIC_by_components.values())
    best_dimensionality_list = [
        key for key in AIC_by_components if AIC_by_components[key] == minimum_AIC]
    # if there happen to be mutiple matches, take the lower dimensionality
    # which is the first in the list since we used an ordered dict
    best_dimensionality = best_dimensionality_list[0]
    print(f'best dimensionality by AIC: {best_dimensionality}')
    return(best_dimensionality)


# compute FA solution for chosen dimensionality
def fit_and_score_factor_analysis(data, n_components):
    fa = FactorAnalysis(n_components)
    scores = fa.fit_transform(data)
    AIC = n_components * 2 - 2 * fa.score(data)

    factor_analysis_result = namedtuple(
        "factor_analysis_result",
        ["loadings", "scores", "AIC"])
    return(factor_analysis_result(fa.components_, scores, AIC))


# print report with highest absolute loading variables for each component
# and report pearson correlation and p-value for correlation between health variable and
# component scores
def create_loading_report_by_component(scores, loadings,
                                       survey_data, health_data,
                                       n_loadings_to_print=3):
    n_components = scores.shape[1]
    variable_names = list(survey_data.columns)

    for component in range(n_components):
        print(f'Component {component}')
        correlation, pvalue = pearsonr(
            scores[:, component], health_data.loc[:, 'mental_health'])
        bonferroni_p = min((1, pvalue * n_components))
        print(f'r = {correlation:.3f}, Bonferroni p = {bonferroni_p:.3f}')

        absolute_loadings = np.abs(loadings[component, :])
        absolute_loading_argsort_descending = np.argsort(absolute_loadings)[::-1]

        for variable_idx in range(n_loadings_to_print):
            print('%s (%0.3f)' % (
                variable_names[absolute_loading_argsort_descending[variable_idx]],
                loadings[component, absolute_loading_argsort_descending[variable_idx]]))
        print('')


# utility function to scale a data frame
def scale_data_frame(df):
    return(pd.DataFrame(scale(df.values), columns=df.columns, index=df.index))


if __name__ == "__main__":
    basedir = Path(os.getcwd())
    datadir = basedir / 'data'

    health_data_raw = load_health_data(datadir)
    health_data_selected = select_wanted_health_variables(health_data_raw)
    health_data_mean = pd.DataFrame(health_data_selected.mean(1),
                                    columns=['mental_health'])

    behavioral_data_raw = load_behavioral_data(datadir)
    survey_data_full = extract_surveys_from_behavioral_data(behavioral_data_raw)

    health_data_nonan, survey_data_nonan = remove_NA_rows_from_matched_data_frames(
        health_data_mean, survey_data_full)

    health_data = scale_data_frame(health_data_nonan)
    survey_data = scale_data_frame(survey_data_nonan)

    confirm_data_frame_index_alignment(health_data, survey_data)

    n_components = get_best_FA_dimensionality(survey_data)
    factor_analysis_result = fit_and_score_factor_analysis(survey_data, n_components)

    create_loading_report_by_component(
        factor_analysis_result.scores,
        factor_analysis_result.loadings,
        survey_data,
        health_data)
