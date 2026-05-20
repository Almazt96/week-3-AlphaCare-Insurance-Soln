import pandas as pd
import numpy as np
import os
import zipfile

def load_insurance_data(file_path: str) -> pd.DataFrame:
    """
    Loads insurance datasets cleanly. Handles streaming compression (.zip), 
    traditional text formats, and sniffs out pipe '|' or standard delimiters.
    """
    try:
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.zip':
            with zipfile.ZipFile(file_path, 'r') as z:
                # Target the first data file discovered inside the compressed archive
                file_list = [f for f in z.namelist() if not f.startswith('__MACOSX') and not f.endswith('/')]
                if not file_list:
                    raise IOError("The provided ZIP file is empty or contains no readable text files.")
                
                target_internal_file = file_list[0]
                
                # Open a stream to read the first line of the zipped file for delimiter sniffing
                with z.open(target_internal_file, 'r') as f_stream:
                    # Read line, decode bytes to string
                    first_line = f_stream.readline().decode('utf-8', errors='ignore')
                
                # Assign delimiter based on structural fingerprinting
                separator = '|' if '|' in first_line else (';' if ';' in first_line else ',')
                
                # Read directly from the compressed stream
                df = pd.read_csv(z.open(target_internal_file), sep=separator)
                
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            
        else:
            # Handle plain uncompressed tabular files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
            separator = '|' if '|' in first_line else (';' if ';' in first_line else ',')
            df = pd.read_csv(file_path, sep=separator)
            
        # Standard Clean: Strip any leading/trailing blank spaces from the headers
        df.columns = df.columns.str.strip()
            
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
            
        return df
        
    except Exception as e:
        raise IOError(f"Failed to process archive or data file at {file_path}. Details: {e}")

def calculate_insurance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes critical domain metrics: Loss Ratio and Profit Margin."""
    df = df.copy()
    
    premium_col = 'TotalPremium' if 'TotalPremium' in df.columns else 'CalculatedPremiumPerTerm'
    claims_col = 'TotalClaims' if 'TotalClaims' in df.columns else 'TotalClaimAmount'
    
    if premium_col not in df.columns or claims_col not in df.columns:
        raise KeyError(
            f"Required risk calculation inputs missing. Found columns: {df.columns.tolist()}"
        )
        
    premium_safe = df[premium_col].replace(0, np.nan)
    df['Loss_Ratio'] = df[claims_col] / premium_safe
    df['Loss_Ratio'] = df['Loss_Ratio'].fillna(0)
    df['Margin'] = df[premium_col] - df[claims_col]
    return df