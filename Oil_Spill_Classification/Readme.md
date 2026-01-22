#  Oil Spill Detection — Machine Learning Classification

This project focuses on detecting oil spills using machine learning classification techniques.  
The dataset presents **high dimensionality, class imbalance, and severe multicollinearity**, making it a strong case study for comparing classical statistical and margin-based models.

---

##  Problem Overview

Oil spill detection is a **critical environmental monitoring task**.  
False negatives can cause severe ecological damage, while false positives lead to unnecessary operational costs.

The key challenges addressed in this project include:
- **Highly correlated features (multicollinearity)**
- **Imbalanced class distribution**
- **Model stability and generalization**

---

##  Dataset

- **Source:** [Kaggle – Oil Spill Detection Dataset](https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection)
- **Samples:** 937
- **Features:** 49 numerical attributes
- **Target:** Binary classification  
  - `1` → Oil Spill  
  - `0` → No Oil Spill  

### Dataset Characteristics
- No missing values
- Mix of integer and continuous variables
- Strong feature correlation → causes instability in some linear models

---

##  Data Preprocessing Pipeline

### 1️⃣ Feature / Target Split
- `X`: 49 input features  
- `y`: target variable

---

### 2️⃣ Handling Class Imbalance — SMOTE

The dataset is **imbalanced**, which biases models toward the majority class.

✔️ **SMOTE (Synthetic Minority Oversampling Technique)** was applied to:
- Generate synthetic minority-class samples
- Improve recall and decision boundary learning

---

### 3️⃣ Train–Test Split
- **80% Training**
- **20% Testing**
- `random_state = 42` for reproducibility

---

### 4️⃣ Feature Scaling
✔️ **StandardScaler** was used to normalize features  
This step is essential for:
- Logistic Regression
- LDA / QDA
- Support Vector Machines

---

##  Models Implemented

Four classification models were trained and evaluated:

### 🔹 Logistic Regression
- Baseline linear probabilistic classifier
- Sensitive to multicollinearity
- Used to establish a performance reference

---

### 🔹 Linear Discriminant Analysis (LDA)
- Assumes shared covariance matrix
- Performs dimensionality reduction
- Struggles under strong multicollinearity

---

### 🔹 Quadratic Discriminant Analysis (QDA)
- Allows **class-specific covariance matrices**
- Better suited for correlated features
- Generated warnings due to collinearity — **expected and informative**

---

### 🔹 Support Vector Machine (SVM)
- Tested with:
  - Linear kernel
  - Polynomial kernel
  - RBF kernel
- Hyperparameter tuning using **GridSearchCV**
- Strong margin maximization → robust under multicollinearity

---

##  Evaluation Metrics

Each model was evaluated using:

- **Accuracy**
- **Precision**
- **Recall (Sensitivity)**
- **Specificity**
- **F1 Score**
- **Confusion Matrices (Train & Test)**
- **ROC Curves & AUC**

This multi-metric approach ensures performance is not misleading due to class imbalance.

---

##  Model Performance Summary

### Key Observations

- **All four models performed well**
- **QDA and SVM consistently outperformed Logistic Regression and LDA**
- SVM with **RBF kernel achieved the highest accuracy (~97–98%)**

### Why QDA & SVM Performed Better
- They **handle multicollinearity more effectively**
- QDA models class-specific variance
- SVM focuses on decision margins rather than feature independence

---

## Important Note on High Accuracy

> **Such high accuracy is not typically observed in real-world oil spill detection systems.**

Reasons:
- The dataset is **clean and structured**
- Synthetic samples were introduced using **SMOTE**
- Features are highly informative and correlated

--> This project emphasizes **model comparison and behavior analysis**, not deployment-ready performance claims.

---

##  ROC Curve Analysis

ROC curves were generated for:
- Logistic Regression
- LDA
- QDA
- SVM

**SVM and QDA achieved the highest AUC values**, indicating strong class separation and robust probability estimates.

---

##  Key Technical Takeaways

- Multicollinearity severely affects linear probabilistic models
- Margin-based and quadratic classifiers are more resilient
- Oversampling improves recall but must be interpreted carefully
- Accuracy alone is insufficient — confusion matrices and ROC curves are critical

---

##  Technologies Used

- **Python**
- **NumPy & Pandas**
- **Scikit-learn**
  - Logistic Regression
  - LDA / QDA
  - SVM
  - GridSearchCV
- **Imbalanced-learn (SMOTE)**
- **Matplotlib**

---

##  Conclusion

This project demonstrates how **different machine learning models respond to multicollinearity and imbalance**.  
It highlights why **model assumptions matter** and why evaluation must go beyond accuracy — especially in high-stakes environmental applications.
