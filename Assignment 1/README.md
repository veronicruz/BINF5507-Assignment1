# BINF5507-Assignment1
Object-Oriented Data Cleaning and Preprocessing
The assigment involves cleaning and preprocessing a messy dataset and prepare the data to be input into a simple model and test the impact of clean data on model performance. Preprocessing steps include imputing missing values with mean, dropping/removing duplicate rows, normalizaing data with min-max scaler or standard scaler, and removing redundant or duplicate column using correlation matrix with corr() function. 
The code is structured in a reusable format, making it suitable for use in real-world data science pipelines or coursework.

Files
data_preprocessor.py: Contains all core preprocessing functions.
main.ipynb: Jupyter notebook to test and demonstrate the functions.
README.md: Project documentation.

Dataset - messy_data.csv
Description: The dataset contains 1196 records with 28 columns including numeric, categorical, and boolean features potentially related to cardiac diagnostics.

Key Features and Columns
target: Binary outcome variable possibly indicating condition presence
a, d, e, g, i, m, s, {: Categorical or boolean columns
b, c, f, h, j ... : Numeric columns

Functions 
1. impute_missing_values(data, strategy='mean'): 
    Purpose: Fills missing values in the dataset with one of the strategies: ('mean', 'median', 'mode').
    Parameters: 
        - pandas DataFrame
        - str, imputation method ('mean', 'median', 'mode')
    Return: pandas DataFrame with missing values imputed 

2. remove_duplicates(data)
    Purpose: Removes duplicate rows from the dataset.
    Parameters: 
        - pandas DataFrame
    Return: pandas DataFrame without duplicates.

3. normalize_data(data, method='minmax')
    Purpose: Apply normalization to numerical features with Min-Max Scaling or Standard Scaling.
    Parameters: 
        - pandas DataFrame
        - str, normalization method ('minmax' (default) or 'standard')
    Return: Scaled pandas DataFrame

4. remove_redundant_features(data, threshold=0.9)
    Purpose: Remove redundant or duplicate columns based on correlation threshold (correlation > threshold).
    Parameters: 
        - pandas DataFrame
        - float, correlation threshold
    Return: pandas DataFrame with highly correlated (redundant) features dropped.

Model Function: simple_model()
- serves as a tool to compare logistic regression model performance before and after data cleaning and preprocessing. 
- Returns model accuracy based on data input (orginal dataset or cleaned dataset).

How to Run
1. Clone the repository
2. Ensure you have Python 3.x and dependencies installed - see below.
3. data_preprocessor.py executes cleaning functions - double check if there are no errors.
4. Run main.ipynb notebook load dataset, to call functions and explore the cleaned dataset. 

Output
 - cleaned dataset with no missing numeric values 
 - removed duplicate rows if needed
 - all numeric data scaled
 - redudant features removed using correlation matrix 

Dependancies 
- pandas
- numpy 
- scikit-learn