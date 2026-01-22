# Logistic Regression for Diabetes Prediction

This project applies **Logistic Regression** to model and predict the probability of diabetes based on clinical and physiological variables. Unlike standard linear models, this approach is tailored for binary outcomes, making it a powerful tool for medical diagnosis and risk assessment.

##  Dataset Overview
The model was trained on data from **314 patients**, focusing on the relationship between physical metrics and diabetic status.

* **Target Variable**: Binary classification (1 = Positive, 0 = Negative).
* **Key Predictors**: 
    * Glucose Concentration (Primary Predictor)
    * Body Mass Index (BMI / Mass)
    * Pregnancy History
    * Blood Pressure, Insulin, Age, and Pedigree.



---

##  Why Logistic Regression?
For medical diagnosis, we need more than just a "Yes/No" answer; we need **probability** and **statistical significance**. Logistic Regression was chosen because it provides:
1.  **Probabilistic Interpretation**: Outputs a risk percentage (0-100%).
2.  **Explainability**: Uses log-odds and coefficients to explain exactly how each feature affects the outcome.
3.  **Statistical Rigor**: Provides $z$-statistics and $p$-values to prove the reliability of each predictor.

---

##  Model Development & Results

### 1. Simple Logistic Regression (Univariate)
**Model**: `diabetes ~ glucose`
* **Result**: Glucose was found to be a highly significant predictor ($p < 0.001$).
* **Insight**: As glucose levels increase, the log-odds of diabetes increase significantly.
* **Predictive Power**: A patient with glucose level 20 has a ~0.5% risk, while a level of 180 results in an **83.6% predicted probability**.

### 2. Multivariate Logistic Regression
**Model**: `diabetes ~ glucose + mass + pregnant`
By combining multiple factors, the model improved its explanatory power (**Pseudo $R^2$ ≈ 0.30**).

| Variable | Interpretation | Significance |
| :--- | :--- | :--- |
| **Glucose** | Strongest positive correlation with risk | $p < 0.001$ |
| **Mass (BMI)** | Higher BMI significantly increases probability | $p < 0.01$ |
| **Pregnancy** | Each pregnancy adds to the cumulative risk | $p < 0.05$ |



---

##  Key Visualizations & Analysis
* **Partial Regression (Effect) Plots**: These plots were used to visualize the isolated impact of each predictor while holding others constant. This confirms that the relationships are not driven by confounding variables.
* **Likelihood Ratio Test**: The multivariate model achieved a $p$-value $\ll 0.001$, confirming it is significantly more accurate than a null model.

---

##  Key Takeaways
* **Glucose** is the most critical independent predictor of diabetes in this dataset.
* **BMI and Pregnancy history** provide essential context that improves the model's accuracy.
* The model successfully balances **statistical rigor** with **clinical relevance**, providing a practical tool for medical risk assessment.

##  Technologies Used
* **Python / R** (Specify which you used)
* **Statsmodels** (for OLS and Logistic Summaries)
* **Matplotlib / Seaborn** (for Effect Plots)
* **Pandas & NumPy**
