import os
import sys
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Importing directly since prepare.py and data_loader.py live in the same directory
import data_loader

def prepare_data():
    print("Starting data preparation...")
    
    # Locate params.yaml relative to this script's directory instead of the terminal root
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Points to 'src'

    project_root = os.path.dirname(script_dir)               # Points to project root

    params_path = os.path.join(project_root, "params.yaml")
    
    # 1. Load parameters from params.yaml
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    
    split_ratio = params["prepare"]["split_ratio"]
    seed = params["prepare"]["seed"]
    
    # 2. Define paths relative to the project root as well so they don't break

    raw_data_path = os.path.join(project_root, "data", "raw", "MachineLearningRating_v3.zip") 
    train_output_path = os.path.join(project_root, "data", "processed", "cleaned_insurance_data.csv")
    test_output_path = os.path.join(project_root, "data", "processed", "test.csv")
       
    # 3. Load the raw data using your custom function
    df = data_loader.load_insurance_data(raw_data_path)
  
    # Optional: If you want to compute the Loss_Ratio and Margin metrics 
    # before splitting, you can uncomment the line below:
    # df = data_loader.calculate_insurance_metrics(df)
    
    # 4. Perform the train/test split
    train_df, test_df = train_test_split(
        df, 
        test_size=split_ratio, 
        random_state=seed
    )
    
    # 5. Save the split datasets back to CSVs
    print(f"Saving prepared data to {train_output_path}")
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    
    print("Data preparation complete!")


# Add this block at the very bottom of the file (no indentation)

# Main execution block

if __name__ == "__main__":
    prepare_data()