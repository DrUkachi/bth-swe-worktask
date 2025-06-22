import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from scipy import stats
warnings.filterwarnings('ignore')

def generate_original_data(num_samples=500, seed=123):
  """
  This function generates the original dataset as provided in the code
  """

  np.random.seed(seed)
  df = pd.DataFrame(
        {
            "Category1": np.random.choice(["A", "B", "C", "D", "E"],
                                         num_samples, p=[0.2, 0.4, 0.2, 0.1, 0.1]),
            "Value1": np.random.normal(10, 2, num_samples),
            "Value2": np.random.normal(20, 6, num_samples),
        }
    )
  return df

def infer_parameters(df):
  """
  Analyses the dataframe to infer its statistical parameters.
  This is the main task: To determine the characteristics of 
  the data without any information about the original parameters
  """
  parameters = {}

  # Inferring the categorical distribution (probabilities of each category)
  parameters['cat1_dist'] = df['Category1'].value_counts(normalize=True).to_dict()

  # Infer continous distribution parameters (mean and standard deviation)
  parameters['val1_mean'] = df['Value1'].mean()
  parameters['val1_std'] = df['Value1'].std()

  parameters['val2_mean'] = df['Value2'].mean()
  parameters['val2_std'] = df['Value2'].std()

  print("Infered parameters:")
  print(f"Category1 Distribution: {parameters['cat1_dist']}")
  print(f"Value1 Mean: {parameters['val1_mean']}, Std: {parameters['val1_std']}")
  print(f"Value2 Mean: {parameters['val2_mean']}, Std: {parameters['val2_std']}")

  return parameters

def generate_new_data(parameters, num_samples, seed):
  """
  Generates new data based on the inferred parameters
  """
  # Unpack the inferred parameters
  cat1_dist = parameters['cat1_dist']
  val1_mean = parameters['val1_mean']
  val1_std = parameters['val1_std']
  val2_mean = parameters['val2_mean']
  val2_std = parameters['val2_std']

  new_df = pd.DataFrame(
      {
      'Category1': np.random.choice(list(cat1_dist.keys()), size=num_samples, p=list(cat1_dist.values())),
      'Value1': np.random.normal(loc=val1_mean, scale=val1_std, size=num_samples),
      'Value2': np.random.normal(loc=val2_mean, scale=val2_std, size=num_samples)
    }
      )

  return new_df

def verify_similarity(original_df, new_df):
    """Performs statistical verification of similarity."""
    print("\n Statistical Verification:")
    
    # Chi-Squared test for Category1
    # This test compares the frequency counts of categories between the two datasets.
    # Null Hypothesis (H0): The distributions are the same.
    # A high p-value (> 0.05) means we fail to reject H0, which is our desired outcome.
    original_counts = original_df['Category1'].value_counts().sort_index()
    new_counts = new_df['Category1'].value_counts().reindex(original_counts.index).fillna(0)
    
    chi2, p_cat = stats.chisquare(f_obs=new_counts, f_exp=original_counts)
    
    print(f"\nChi-Squared Test for 'Category1':")
    print(f"  This test checks if the categorical distributions are the same.")
    print(f"  Statistic: {chi2:.4f}, P-value: {p_cat:.4f}")
    if p_cat > 0.05:
        print("  Result: The distributions are not significantly different (Good).")
    else:
        print("  Result: The distributions are significantly different (Bad).")
    
    # KS test for continuous variables
    # This test checks if two continuous samples come from the same distribution.
    # Null Hypothesis (H0): The two samples are from the same distribution.
    # A high p-value (> 0.05) is our desired outcome.
    for col in ['Value1', 'Value2']:
        ks_stat, p_val = stats.ks_2samp(original_df[col], new_df[col])
        print(f"\nKolmogorov-Smirnov Test for '{col}':")
        print(f"  This test checks if the continuous distributions are the same.")
        print(f"  Statistic: {ks_stat:.4f}, P-value: {p_val:.4f}")
        if p_val > 0.05:
            print("  Result: The distributions are not significantly different (Good).")
        else:
            print("  Result: The distributions are significantly different (Bad).")


def visual_verification(original_df, new_df):
  """
    Creates and displays plots to visually compare the two datasets.
    
    - A grouped bar chart for the categorical column.
    - Overlaid Kernel Density Plots for the continuous columns.
  """
  print("\nGenerating Visual Verification Plots:")

  # Set a nice style for the plots
  sns.set_style("whitegrid")
  # Create a figure with 3 subplots in a single column
  fig, axes = plt.subplots(3, 1, figsize=(10, 18))
  fig.suptitle('Visual Comparison of Original and New Datasets', 
               fontsize=16, y=0.95)
  # --- Plot 1: Categorical Data Comparison ---
  # To create a grouped bar chart, we first combine the dataframes
  # and add a 'Source' column to distinguish them.
  
  original_df['Source'] = 'Original'
  new_df['Source'] = 'New'
  combined_df = pd.concat([original_df, new_df])

  sns.countplot(
        data=combined_df, 
        x='Category1', 
        hue='Source', 
        ax=axes[0],
        order=['A', 'B', 'C', 'D', 'E'] # Ensure consistent ordering
    )
  
  axes[0].set_title('Comparison of Categorical Distribution for "Category1"', fontsize=14)
  axes[0].set_xlabel('Category', fontsize=12)
  axes[0].set_ylabel('Count', fontsize=12)

  # --- Plot 2: Continuous Data Comparison for "Value1" ---
  sns.kdeplot(data=original_df, x='Value1', fill=True, alpha=0.5, label='Original', ax=axes[1])
  sns.kdeplot(data=new_df, x='Value1', fill=True, alpha=0.5, label='New', ax=axes[1])
  axes[1].set_title('Comparison of Continuous Distribution for "Value1"', fontsize=14)
  axes[1].set_xlabel('Value1', fontsize=12)
  axes[1].set_ylabel('Density', fontsize=12)
  axes[1].legend()

  # --- Plot 3: Continuous Data Comparison for "Value2" ---
  sns.kdeplot(data=original_df, x='Value2', fill=True, alpha=0.5, label='Original', ax=axes[2])
  sns.kdeplot(data=new_df, x='Value2', fill=True, alpha=0.5, label='New', ax=axes[2])
  axes[2].set_title('Comparison of Continuous Distribution for "Value2"', fontsize=14)
  axes[2].set_xlabel('Value2', fontsize=12)
  axes[2].set_ylabel('Density', fontsize=12)
  axes[2].legend()

  # Clean up the layout and display the plots
  plt.tight_layout(rect=[0, 0, 1, 0.96])
  plt.savefig('bth-phd-sweden/visual_verification.png')
  plt.show()

if __name__ == "__main__":

    NUM_SAMPLES = 10000
    SEED = 123
    # Generate the original dataset
    original_df = generate_original_data(NUM_SAMPLES, SEED)
    
    # Infer parameters from the original dataset
    inferred_parameters = infer_parameters(original_df)
    
    # Generate new data based on the inferred parameters
    new_df = generate_new_data(inferred_parameters, NUM_SAMPLES, SEED)

    # Statistical verification of the original and new datasets
    verify_similarity(original_df.copy(), new_df.copy())
    
    # Visual verification of the original and new datasets
    visual_verification(original_df, new_df)
