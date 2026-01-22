Logistic Regression for Diabetes Prediction

This project applies logistic regression to model the probability of diabetes based on clinical and physiological variables. Unlike linear regression, logistic regression is specifically designed for binary outcomes, making it suitable for medical diagnosis tasks.

Dataset Overview

Observations: 314 patients

Target Variable:

diabetes = 1 → Positive

diabetes = 0 → Negative

Predictors include:

Glucose concentration

Body Mass Index (BMI / mass)

Number of pregnancies

Blood pressure, insulin, age, pedigree, etc.

The response variable was converted from categorical (pos, neg) to a binary numeric format to allow logistic modeling.

Why Logistic Regression?

Medical diagnosis problems require:

Probabilistic interpretation

Statistical significance testing

Explainability

Logistic regression provides:

Interpretable coefficients (log-odds)

Confidence intervals

Hypothesis testing using z-statistics and p-values

This makes it especially valuable in healthcare contexts.

Simple Logistic Regression (Univariate Model)
Model:
diabetes ~ glucose

Key Results:

Glucose coefficient > 0

Highly significant (p < 0.001)

Pseudo R² ≈ 0.23

Interpretation:

As glucose levels increase, the log-odds of diabetes increase

Glucose alone explains a meaningful portion of diabetes risk

The low p-value confirms glucose is a strong independent predictor

Practical Meaning:

A patient with very low glucose has a near-zero predicted probability

High glucose values dramatically increase predicted risk

Probability Prediction (Clinical Interpretation)

The model outputs probabilities, not just class labels.

Example:

Glucose = 20 → Probability ≈ 0.5% → Non-diabetic

Glucose = 180 → Probability ≈ 83.6% → Diabetic

This probabilistic output is crucial for risk-based medical decisions, not just yes/no classification.

Multivariate Logistic Regression
Model:
diabetes ~ glucose + mass + pregnant

Why Multivariate?

Diabetes is multifactorial. Using multiple predictors:

Controls for confounding variables

Improves model fit

Produces more realistic risk estimates

Model Performance & Statistical Strength

Pseudo R² ≈ 0.30 → improved explanatory power

All predictors are statistically significant

Likelihood Ratio Test p-value ≪ 0.001

This confirms the multivariate model is significantly better than a null model.

Coefficient Interpretation (Very Important)
Variable	Interpretation
Glucose	Higher glucose → strong increase in diabetes risk
Mass (BMI)	Higher BMI → higher probability of diabetes
Pregnant	Each additional pregnancy increases diabetes risk

All effects are positive and statistically significant, aligning with medical literature.

Partial Regression (Effect) Plots

The effect plots visualize the isolated impact of each predictor while controlling for others.

What These Plots Show:

Each dot represents an observation

The line shows the adjusted relationship

Other predictors are held constant

Why This Matters:

Confirms relationships are not driven by confounding

Shows monotonic risk increase

Strengthens interpretability and trust in the model

Key Takeaways

Logistic regression provides interpretable medical insights

Glucose is the strongest single predictor

BMI and pregnancy history add meaningful predictive power

Partial regression plots validate independent effects

The model balances statistical rigor and clinical relevance
