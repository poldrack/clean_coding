# Refactoring Example 1: pseudocode
# describe the steps that we need to implement, as comments


# load health data

# remove unwanted variables from health data, keeping only mental health scales

# load behavioral data (which includes survey and task data)

# extract survey data from behavioral data
# survey variables are labeled <survey_name>_survey.<variable name>
# so filter on variables that include "_survey" in their name

# align datasets, ensuring that order of subjects is the same

# remove subjects with missing data for either set of variables

# perform FA across a range of n_components and choose dimensionality with mimimum AIC

# compute FA solution for chosen dimensionality

# print report with highest absolute loading variables for each component
# and report pearson correlation and p-value for correlation between health variable and
# component scores