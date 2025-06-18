# BINF5507- Assignment 2
Regression and Classification Models
This assignment involves training and evaluating machine learning models for regression (predicting cholestrol levels) and classification (predicting presence of heart disease). The analysis includes interpreting hyperparameter effects on model performance. Results for linear regression models are  are vizualized with heatmaps of l1_ratio and alpha combinations. ROC and precision-recall curves are used for viewing classification model effectiveness. The code is written in a jupyter notebook split into the following sections: 
- Import libraries 
- Data Preparation: Data Cleaning and Feature Engineering  
- Linear Regression Model: ElasticNet
- Classification Models: Logistic Regression vs. K-Nearest Neighbours (k-NN)

Files 
data/heart_disease_uci(1): Dataset for model training and testing.
scripts/main.ipynb: Jupyter notebook which contains all steps in this assignment
README.md: Project documentation.

Dataset- heart_disease_uci(1)
Description: Heart Disease Dataset from the UCI Machine Learning Repository. Features include age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, and more.
Targets:
For regression: 'chol' - cholesterol levels (numeric, continous)
For classification: 'num' - presence of heart disease 
                             - orginally multiclass range from 0 - 4, where 0 is no disease and 1-4 indicates severity of disease presence
                             - converted to binary classification for project objectives  

Key Code Components and Functions 
1. Data Preparation: Data Cleaning and Feature Engineering
 - def dropMissing(data: pd.DataFrame, threshold = 0.15)
    Purpose: Removes columns if they are above threshold as it indicats high missingness. The default threshold is 15%.
 - fillMissing(data: pd.DataFrame)
    Purpose: Imputes missing values with median for numeric columns and 'unknown' for categorical columns
 - One-hot encoding with pd.get_dummies() for categorical variables before modeling 
 - Adjustments of target variables and creatation of specific datasets for each

2. Linear Regression Model: ElasticNet
 - Uses ElasticNet from sklearn.linear_model for predicting cholestrol 
 - creates a grid and does a gridsearch for l1_ratio and alpha values
 - Evaluation metrics: Root Mean Squared Error (RMSE) and R^2 score
 - Identifies best preforming (optimized) configurations 
 - Results for RMSE and R^2 visualized with heatmaps 

3. Classification Models: Logistic Regression vs. K-Nearest Neighbours (k-NN)
 - Use Logistic Regression model and k-NN model from sklearn.linear_model and sklearn.neighbors resepectively 
 - Logistic Regression grid search on solver and penalty  
 - k-NN grid search on n_neighbors and distance metric
 - Hyperparameter tuning with GridSearchCV with cv = 5 
 - Evaluation Metrics: accuracy, precision, recall, f1 score, auroc, auprc

4. Visualizations
 - plot_curves(tpr, fpr, auroc, precision, recall, auprc, model_name, minority_class=0.1) 
    Purpose: Generates side by side subplots for AUROC and AUPRC for the given model with the scores
 - ROC curve plots TPR vs FPR
 - Precision-recall curve plots plots precision vs recall 


How to Run
1. Clone the repository
2. Ensure you have Python 3.x and dependencies installed - see below.
3. Run main.ipynb notebook and run each cell in order its presented
4. Explore the outputs

Output
- Cleaned and standardized datasets for cholestrol and heart disease. Each are split into training and testing subsets.
- Trained ElasticNet regression model with hyperparameter tuning and evaluation metrics (R², RMSE).
- Trained Logistic Regression and k-NN classification models with hyperparameter tuning and evaluation metrics (accuracy, F1, AUROC, AUPRC).
- Visualizations: Heatmaps, AUROC curve, AUPRC curve for model comparisons.

Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn