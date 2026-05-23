import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a formatted dataframe detailing counts and rates of null elements."""
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    summary = pd.DataFrame({'Missing Count': missing_count, 'Percentage (%)': missing_pct})
    return summary[summary['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

def plot_loss_ratio_by_dimension(df: pd.DataFrame, dimension: str, save_path: str = None):
    """Generates a structured visualization mapping portfolio performance across risk elements."""
    plt.figure(figsize=(10, 5))
    grouped = df.groupby(dimension)['Loss_Ratio'].mean().reset_index().sort_values(by='Loss_Ratio', ascending=False)

    sns.barplot(data=grouped, x=dimension, y='Loss_Ratio', hue=dimension, palette='viridis', legend=False)
    plt.title(f'Average Loss Ratio distribution by {dimension}')
    plt.xticks(rotation=45)
    plt.ylabel('Loss Ratio (Claims / Premium)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

# 1. Bivariate Scatter Plot (with ZipCode Hue)
# A scatter plot allows you to see if higher premiums correlate with higher claims and helps identify geographic clusters or distinct risk profiles per ZIP code.

# Pro Tip: If your dataset has hundreds of unique ZIP codes, the plot will become cluttered. It's usually best to filter for the top 5–10 most frequent ZIP codes to keep the visualization clear and readable, as shown below.

import matplotlib.pyplot as plt
import seaborn as sns

def plot_premium_vs_claims(df, top_n_zipcodes=5):
    """
    Plots a scatter plot of TotalPremium vs TotalClaims, colored by the top N ZipCodes.
    """
    plt.figure(figsize=(10, 6))
    
    # Filter for top N zip codes to avoid legend clutter
    top_zips = df['ZipCode'].value_counts().nlargest(top_n_zipcodes).index
    filtered_df = df[df['ZipCode'].isin(top_zips)].copy()
    
    # Ensure ZipCode is treated as a discrete category for coloring
    filtered_df['ZipCode'] = filtered_df['ZipCode'].astype(str)
    
    sns.scatterplot(
        data=filtered_df, 
        x='TotalPremium', 
        y='TotalClaims', 
        hue='ZipCode', 
        palette='Set2', 
        alpha=0.7
    )
    
    plt.title(f'TotalPremium vs TotalClaims for Top {top_n_zipcodes} ZipCodes')
    plt.xlabel('Total Premium')
    plt.ylabel('Total Claims')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    # 2. Multivariate Correlation Matrix & HeatmapTo see how these variables interact mathematically across the entire dataset, you can build a Pearson correlation matrix. Since ZipCode is a categorical identifier, we include numeric insurance metrics alongside it to spot wider patterns.
    
def plot_correlation_matrix(df):
    # """
    # Computes and plots a correlation matrix for the key numerical insurance metrics.
    # """
    plt.figure(figsize=(8, 6))
    
    # Focus on the core continuous risk tracking features
    numerical_cols = ['TotalPremium', 'TotalClaims']
    
    # Optional: If you have calculated a localized metric like Loss_Ratio, include it!
    if 'Loss_Ratio' in df.columns:
        numerical_cols.append('Loss_Ratio')
        
    corr_matrix = df[numerical_cols].corr()
    
    # Generate an annotated heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=0.5, 
        vmin=-1, 
        vmax=1
    )
    
    plt.title('Correlation Matrix: Premiums vs. Claims')
    plt.show()
    
#     3. Advanced Step: Direct Per-ZipCode Correlations
# If you want to see exactly how tightly bound premiums and claims are inside each individual area, you can group by ZipCode and calculate the linear correlation coefficient for each pocket of data:
    
def get_zipcode_correlations(df, min_samples=30):
    # """
    # Calculates the Pearson correlation coefficient between TotalPremium and TotalClaims 
    # individually for each ZipCode that has an adequate sample size.
    # """
    # Filter out zip codes with too few entries to avoid statistical noise
    zip_counts = df['ZipCode'].value_counts()
    valid_zips = zip_counts[zip_counts >= min_samples].index
    filtered_df = df[df['ZipCode'].isin(valid_zips)]
    
    # Calculate the correlation for each group
    zip_corrs = (
        filtered_df.groupby('ZipCode')
        .apply(lambda g: g['TotalPremium'].corr(g['TotalClaims']))
        .reset_index(name='Premium_Claims_Correlation')
    )
    
    # Sort by strongest positive relationship
    return zip_corrs.sort_values(by='Premium_Claims_Correlation', ascending=False)

# Usage example:
# zip_analysis_df = get_zipcode_correlations(your_dataframe)
# print(zip_analysis_df.head())

# What to look for in your results:High Positive Correlation ($\approx 1.0$): In these ZIP codes, premium scaling is heavily and predictably tied directly to actual claim costs.Low or Zero Correlation ($\approx 0.0$): Indicates high variance or potential mispricing anomalies in specific neighborhoods—places where high premiums don't necessarily map to high claims, or vice versa.

import matplotlib.pyplot as plt
import seaborn as sns

def plot_premium_vs_claims(df, top_n_zipcodes=5):
    """
    Plots a scatter plot of TotalPremium vs TotalClaims, colored by the top N ZipCodes.
    """
    plt.figure(figsize=(10, 6))
    
    # Filter for top N zip codes to avoid legend clutter
    top_zips = df['ZipCode'].value_counts().nlargest(top_n_zipcodes).index
    filtered_df = df[df['ZipCode'].isin(top_zips)].copy()
    
    # Ensure ZipCode is treated as a discrete category for coloring
    filtered_df['ZipCode'] = filtered_df['ZipCode'].astype(str)
    
    sns.scatterplot(
        data=filtered_df, 
        x='TotalPremium', 
        y='TotalClaims', 
        hue='ZipCode', 
        palette='Set2', 
        alpha=0.7
    )
    
    plt.title(f'TotalPremium vs TotalClaims for Top {top_n_zipcodes} ZipCodes')
    plt.xlabel('Total Premium')
    plt.ylabel('Total Claims')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
# 2. Multivariate Correlation Matrix & Heatmap
# To see how these variables interact mathematically across the entire dataset, you can build a Pearson correlation matrix. Since ZipCode is a categorical identifier, we include numeric insurance metrics alongside it to spot wider patterns.

def plot_correlation_matrix(df):
    """
    Computes and plots a correlation matrix for the key numerical insurance metrics.
    """
    plt.figure(figsize=(8, 6))
    
    # Focus on the core continuous risk tracking features
    numerical_cols = ['TotalPremium', 'TotalClaims']
    
    # Optional: If you have calculated a localized metric like Loss_Ratio, include it!
    if 'Loss_Ratio' in df.columns:
        numerical_cols.append('Loss_Ratio')
        
    corr_matrix = df[numerical_cols].corr()
    
    # Generate an annotated heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=0.5, 
        vmin=-1, 
        vmax=1
    )
    
    plt.title('Correlation Matrix: Premiums vs. Claims')
    plt.show()
# 3. Advanced Step: Direct Per-ZipCode Correlations

def get_zipcode_correlations(df, min_samples=30):
    """
    Calculates the Pearson correlation coefficient between TotalPremium and TotalClaims 
    individually for each ZipCode that has an adequate sample size.
    """
    # Filter out zip codes with too few entries to avoid statistical noise
    zip_counts = df['ZipCode'].value_counts()
    valid_zips = zip_counts[zip_counts >= min_samples].index
    filtered_df = df[df['ZipCode'].isin(valid_zips)]
    
    # Calculate the correlation for each group
    zip_corrs = (
        filtered_df.groupby('ZipCode')
        .apply(lambda g: g['TotalPremium'].corr(g['TotalClaims']))
        .reset_index(name='Premium_Claims_Correlation')
    )
    
    # Sort by strongest positive relationship
    return zip_corrs.sort_values(by='Premium_Claims_Correlation', ascending=False)

# Usage example:
# zip_analysis_df = get_zipcode_correlations(your_dataframe)
# print(zip_analysis_df.head())

# What to look for in your results:
# High Positive Correlation (≈1.0): In these ZIP codes, premium scaling is heavily and predictably tied directly to actual claim costs.

# Low or Zero Correlation (≈0.0): Indicates high variance or potential mispricing anomalies in specific neighborhoods—places where high premiums don't necessarily map to high claims, or vice versa.

# show how to do "Geographic Trends - compare cover type, premium, and auto make across provinces"

# To effectively compare features like Cover Type, Premium, and Auto Make across different geographic Provinces, we have to blend categorical distributions with numerical aggregations.

# Because we are dealing with a mix of continuous data (TotalPremium) and high-cardinality categorical variables (VehicleMake or CoverType), the best approach uses clean Pandas group operations combined with targeted visualizations.

# 1. Comparing Premium Across Provinces (Numerical vs. Category)
# To see if certain provinces carry structurally higher risks (and thus higher premiums), a Boxplot combined with an aggregated mean comparison table works best. Boxplots show the median, spread, and any potential premium outliers per region.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_premium_by_province(df):
    """Visualizes premium distribution and calculates average premium per province."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    # Left Side: Boxplot to identify variation, range, and outlier policies
    sns.boxplot(data=df, x='Province', y='TotalPremium', ax=axes[0], palette='viridis', hue='Province', legend=False)
    axes[0].set_title('Premium Distribution Across Provinces')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Right Side: Clean summary table of averages
    summary = df.groupby('Province')['TotalPremium'].agg(['mean', 'median', 'count']).reset_index()
    summary = summary.sort_values(by='mean', ascending=False)
    
    # Render table values onto the plot axes cleanly
    axes[1].axis('off')
    table = axes[1].table(cellText=summary.values, colLabels=summary.columns, loc='center', cellLoc='center')
    table.scale(1.2, 1.8)
    axes[1].set_title('Summary Metrics by Region', pad=20)
    
    plt.tight_layout()
    plt.show()
