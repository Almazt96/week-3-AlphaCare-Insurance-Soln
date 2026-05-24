import pandas as pd
import numpy as np
from scipy import stats

# --- Define the Missing Statistical Test Functions ---

def run_categorical_chi2_test(dataframe, col1, col2):
    """
    Performs a Chi-Square test of independence between two categorical variables.
    """
    # Create a contingency table (cross-tabulation)
    contingency_table = pd.crosstab(dataframe[col1], dataframe[col2])
    
    # chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    _, p_value, _, _ = stats.chi2_contingency(contingency_table)
    return _, p_value

def run_numerical_ttest(dataframe, group_col, group1, group2, target_numerical_col):
    """
    Performs a Welch's t-test (unequal variances) comparing exactly two specified 
    groups within a column against a target numerical feature.
    """
    # Filter the target numerical column for both specific groups
    data_g1 = dataframe[dataframe[group_col] == group1][target_numerical_col]
    data_g2 = dataframe[dataframe[group_col] == group2][target_numerical_col]
    
    # Drop NaNs to prevent the test from returning NaN values
    data_g1 = data_g1.dropna()
    data_g2 = data_g2.dropna()
    
    # Handle edge case where a group has no data points left
    if len(data_g1) == 0 or len(data_g2) == 0:
        return np.nan, np.nan

    # equal_var=False runs Welch's t-test instead of standard Student's t-test
    t_stat, p_value = stats.ttest_ind(data_g1, data_g2, equal_var=False)
    return t_stat, p_value
