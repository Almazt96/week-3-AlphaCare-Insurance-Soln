# tests/test_data_loader.py
import pytest
import pandas as pd
import numpy as np
# Assuming you have a src/data_preprocessor.py with a clean_data function
# from src.data_preprocessor import clean_data 

# Simple dummy function example if you're testing standard loading
def simple_feature_prep(df):
    df['Claim_log'] = np.log1p(df['Totalclaims'])  # Example transformation
    return df

def test_simple_feature_prep():
    # 1. Arrange: Create a minimal mock DataFrame
    mock_data = pd.DataFrame({'Totalclaims': [1000, 2000, 3000]})

    # 2. Act: Run your pipeline function
    result_df = simple_feature_prep(mock_data)

    # 3. Assert: Verify it behaves as expected
    assert 'Claim_log' in result_df.columns
    assert not result_df['Claim_log'].isnull().any()