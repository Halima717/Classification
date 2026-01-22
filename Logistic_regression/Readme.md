# House Price Prediction & Regression Analysis

This project builds a comprehensive machine learning pipeline to predict real-world housing prices. The goal was to compare various regression techniques and evaluate how different preprocessing and regularization methods impact predictive accuracy.

##  Project Summary & Objective
Using a real-world housing dataset sourced from **Kaggle** [Link here], the objective was to predict house prices based on physical property features. The project follows a rigorous data science workflow—from cleaning and exploratory analysis to advanced statistical modeling.



---

##  Work Completed

### 1. Data Preprocessing & Feature Engineering
To ensure high model reliability, I performed the following:
* **Data Cleaning**: Handled missing values and removed irrelevant or redundant columns that didn't contribute to price variance.
* **Encoding**: Applied **Label Encoding** to transform categorical variables into a numerical format suitable for regression.
* **Outlier Management**: Detected and removed statistical outliers to prevent model distortion.
* **Normalization**: Applied **Log Transformation** to skewed features to ensure they meet the normality assumptions of linear models.
* **EDA**: Conducted Exploratory Data Analysis with visualizations to understand correlations between property features and price.

### 2. Modeling & Evaluation
I implemented and compared five distinct regression architectures to find the most robust solution:
* **Linear Regression**: Established the baseline performance.
* **Ridge & Lasso**: Applied L2 and L1 regularization to study how penalizing complexity reduces overfitting.
* **Principal Component Regression (PCR)**: Used dimensionality reduction to handle multicollinearity.
* **Partial Least Squares (PLS)**: Optimized the relationship between features and the target variable.



### 3. Regularization & Statistical Analysis
* **Cross-Validation**: Used $k$-fold cross-validation to ensure the results were consistent and generalizable.
* **Alpha Analysis**: Analyzed Ridge and Lasso at varying **alpha values** to find the optimal balance between bias and variance.
* **OLS Summary**: Utilized **Ordinary Least Squares (OLS)** regression summaries to evaluate $p$-values and the significance of individual features.

---

##  Key Outcomes
* **High Performance**: Developed a robust and optimized model that accurately predicts house prices.
* **Insights**: Successfully identified the most influential features affecting property value through regularization.
* **Demonstrated Skills**: This project showcases expertise in **Data Preprocessing**, **Regression Modeling**, **Statistical Visualization**, and **Model Comparison**.

##  Technologies Used
* **Python**
* **Scikit-Learn** (Linear, Ridge, Lasso, PCR, PLS)
* **Pandas & NumPy**
* **Matplotlib & Seaborn**
* **Statsmodels** (for OLS statistical summaries)
