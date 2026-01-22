# Classification Projects Portfolio 

This repository contains **four end-to-end classification projects**, each addressing a **different data type and learning challenge**.  
Together, they demonstrate my ability to apply the *right classification technique* based on the *nature of the data* — images, medical scans, and structured tabular datasets.

---

##  Projects Overview

| Project | Type | Dataset | Technique |
|------|------|--------|----------|
| Bear vs Panda | Binary Image Classification | Wildlife Images | Transfer Learning (VGG16) |
| Brain Tumor MRI | Multi-Class Image Classification | Medical MRI Scans | CNN |
| Oil Spill Detection | Categorical Classification | Tabular Sensor Data | ML Classifiers |
| Diabetes Prediction | Binary Classification | Tabular Medical Data | Logistic Regression |

---

##  1. Bear vs Panda Image Classification  
**(Binary Image Classification – Wildlife Dataset)**

### Problem  
Wildlife image classification is challenging due to:
- Background noise
- Limited labeled data
- Variations in pose and lighting

This project focuses on distinguishing **bears vs pandas** using deep learning.

### Approach  
- Used **VGG16 pre-trained on ImageNet**
- Applied **data augmentation** to improve generalization
- Added custom fully connected layers for binary prediction

### Model Details  
- Architecture: `VGG16 → Flatten → Dense(128) → Dropout → Sigmoid`
- Loss Function: Binary Cross-Entropy
- Optimizer: Adam

### Outcome  
- High training and validation accuracy
- Demonstrates effective use of **transfer learning** on small datasets

---

##  2. Brain Tumor Classification Using MRI Images  
**(Multi-Class Image Classification – Medical Imaging)**

### Problem  
Manual brain tumor diagnosis from MRI scans:
- Is time-consuming
- Requires expert interpretation
- Is prone to human error

This project automates tumor classification into:
- Glioma  
- Meningioma  
- Pituitary Tumor  
- No Tumor  

### Approach  
- Built a **custom CNN** using TensorFlow/Keras
- Used **Batch Normalization** and **Dropout** to stabilize training
- Applied **Early Stopping** to prevent overfitting

### Performance & Evaluation  
- Achieved **81% training accuracy at the 12th epoch**
- Final evaluation was performed on a **separate Test Set**
- Confusion matrices were generated to analyze class-wise performance

> Training accuracy reflects learning progress,  
> while the confusion matrix represents **final generalization performance**

### Why This Matters  
- Shows ability to handle **medical image data**
- Balances performance with interpretability
- Addresses real-world healthcare challenges

---

##  3. Oil Spill Classification  
**(Categorical Classification – Structured / Tabular Data)**

### Problem  
Detecting oil spills from sensor data is critical for:
- Environmental protection
- Early disaster response

However, sensor datasets often contain:
- Noise
- Redundant features
- Class imbalance

### Approach  
- Performed data cleaning and preprocessing
- Applied multiple classification algorithms
- Compared model performance using standard metrics

### Key Focus  
- Handling **categorical structured data**
- Feature relevance and preprocessing
- Model comparison and evaluation

### Outcome  
- Built a reliable classifier for oil spill detection
- Demonstrates versatility beyond image-based models

---

##  4. Diabetes Prediction Using Logistic Regression  
**(Binary Classification – Medical Tabular Data)**

### Problem  
Diabetes diagnosis requires:
- Explainability
- Statistical confidence
- Probability-based risk assessment

Black-box models are often unsuitable in medical contexts.

### Approach  
- Applied **Logistic Regression**
- Built both **univariate** and **multivariate** models
- Used statistical inference (p-values, confidence intervals)

### Key Insights  
- Glucose is the strongest predictor of diabetes
- BMI and pregnancy history significantly improve predictions
- Partial regression plots validate independent variable effects

### Why Logistic Regression?  
- Outputs probabilities, not just labels
- Highly interpretable
- Clinically meaningful and statistically sound

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Statsmodels  
- Pandas & NumPy  
- Matplotlib & Seaborn  

---

##  Key Takeaway

This repository highlights my ability to:
- Choose **appropriate classification techniques**
- Work across **image, medical, and structured datasets**
- Balance **performance, interpretability, and real-world relevance**

Each project reflects a different classification challenge — together forming a **well-rounded machine learning portfolio**.

---

 If you found this useful, feel free to explore the individual project folders for detailed implementations.
