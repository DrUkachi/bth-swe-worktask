# Work Task for PhD Position @ BTH - Coding Task Solution

This repository contains the solution to the coding task provided as part of the work task for the PhD positions at the Software Engineering Department @ BTH.

## 1. Problem Statement

The goal is to analyze a given dataset, infer its underlying statistical properties, and then use those inferred properties to generate a new, similar dataset. A key constraint is to **not** use the original, hardcoded sampling parameters for the new generation. Finally, the similarity between the original and new datasets must be verified.

The provided script `coding_task.py` accomplishes this entire pipeline.

## 2. Methodology

The approach is divided into four main steps, demonstrating a complete cycle of analysis, synthesis, and verification.

### Step 1: Generate the Original Dataset
The `generate_original_data()` function is used to create the initial dataset based on the parameters provided in the work task. This serves as our ground truth for analysis and comparison.

### Step 2: Infer Statistical Parameters
The `infer_parameters()` function loads the original dataset and calculates its statistical characteristics. This is the core of the analysis phase.
- **For the `Category1` column (Categorical Data):** We calculate the frequency distribution of its values (`A`, `B`, `C`, `D`, `E`) to infer the probability of each category occurring.
- **For the `Value1` and `Value2` columns (Continuous Data):** We calculate the **mean** and **standard deviation** for each column, assuming they follow a normal (Gaussian) distribution as suggested by the `numpy.random.normal` function used in the original generation code.

### Step 3: Synthesize New, Similar Dataset
The `generate_new_data()` function uses the **inferred parameters** from Step 2 to generate a new dataset of the same size.
- `Category1` is generated using `numpy.random.choice`, with the inferred probabilities.
- `Value1` and `Value2` are generated using `numpy.random.normal`, with the inferred mean and standard deviation.

This process ensures that the new dataset is statistically modeled on the original, without using the original hardcoded parameters, thus fulfilling the main requirement of the task.

### Step 4: Verify Similarity
This is the most critical step. To prove that the new dataset is "similar" to the original, both statistical tests and visual comparisons were carried out:
*   **Statistical Verification:**
      - **Chi-Squared Goodness-of-Fit Test:** To compare the distribution of the categorical column (`Category1`) between the two datasets.
      - **Two-Sample Kolmogorov-Smirnov (KS) Test:** To compare the distributions of the continuous columns (`Value1` and `Value2`). This test is ideal as it checks if two samples are drawn from the same underlying distribution.

*  **Visual Verification:**
    - **Categorical Data (`Category1`):** A grouped bar chart is generated to visually compare the counts of each category in the original versus the new dataset.
    - **Continuous Data (`Value1`, `Value2`):** Overlaid Kernel Density Estimate (KDE) plots are used to compare the distributions. If the datasets are similar, these curves will closely overlap, indicating a similar shape, mean, and variance.

## 3. Prerequisites

To run this script, you will need Python 3.8+ and the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`

You can install these dependencies using pip:
```bash
pip install pandas numpy matplotlib seaborn scipy
```

## 4. Usage

1. Clone this repository to your local machine.
2. Navigate to the repository's root directory.
3. Run the script from your terminal:
```bash
python coding_task.py
```

## 5. Results & Verification

Running the script will first print the inferred parameters to the console and then display a plot for visual comparison.

### Console Output (Example)

```
Infered parameters:
Category1 Distribution: {'B': 0.3921, 'A': 0.2064, 'C': 0.2058, 'D': 0.1003, 'E': 0.0954}
Value1 Mean: 10.037346621290451, Std: 2.0149842530764968
Value2 Mean: 19.903824406779698, Std: 5.972238085924904

Generated a new dataset with 10000 samples.


Statistical Verification:

Chi-Squared Test for 'Category1':
  This test checks if the categorical distributions are the same.
  Statistic: 4.3312, P-value: 0.3630
  Result: The distributions are not significantly different (Good).

Kolmogorov-Smirnov Test for 'Value1':
  This test checks if the continuous distributions are the same.
  Statistic: 0.0107, P-value: 0.6161
  Result: The distributions are not significantly different (Good).

Kolmogorov-Smirnov Test for 'Value2':
  This test checks if the continuous distributions are the same.
  Statistic: 0.0073, P-value: 0.9527
  Result: The distributions are not significantly different (Good).

--- Generating Visual Verification Plots ---

```

### Visual Output

The script will generate a window containing three plots. The close alignment between the "Original" and "New" data in all three plots provides strong visual evidence of similarity.


![Visual Comparison of Datasets](https://github.com/DrUkachi/bth-swe-worktask/blob/main/visual_verification.png) 


## 6. Conclusion

The methodology successfully generated a statistically similar dataset by first inferring the distribution parameters from the original data and then synthesizing new data based on those inferences. The visual verification step clearly demonstrates that the distributions of the categorical and continuous variables in the new dataset closely match those of the original, confirming the success of the task.