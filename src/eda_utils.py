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
    sns.barplot(data=grouped, x=dimension, y='Loss_Ratio', palette='viridis')
    plt.title(f'Average Loss Ratio distribution by {dimension}')
    plt.xticks(rotation=45)
    plt.ylabel('Loss Ratio (Claims / Premium)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
