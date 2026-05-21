import os
import sys
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    print("Starting data preparation...")
    
    # Locate params.yaml relative to this script's directory instead of the terminal root
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Points to 'src'
    project_root = os.path.dirname(script_dir)               # Points to 'week 3 insurance-risk-analytics'
    params_path = os.path.join(project_root, "params.yaml")
    
    # 1. Load parameters from params.yaml
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    
    split_ratio = params["prepare"]["split_ratio"]
    seed = params["prepare"]["seed"]
    
    # 2. Define paths relative to the project root as well so they don't break
    raw_data_path = os.path.join(project_root, "data", "MachineLearningRating_v3.zip") 
    train_output_path = os.path.join(project_root, "data", "prepared.csv")
    test_output_path = os.path.join(project_root, "data", "test.csv")
       
    # ... rest of your code remains the same ...
    # ... rest of your code remains the same ...
    
    # Make sure you save the files at the end of your function!
    print(f"Saving prepared data to {train_output_path}")
    df_train.to_csv(train_output_path, index=False)
    df_test.to_csv(test_output_path, index=False)
    print("Data preparation complete!")

# Add this block at the very bottom of the file (no indentation)
if __name__ == "__main__":
    prepare_data()