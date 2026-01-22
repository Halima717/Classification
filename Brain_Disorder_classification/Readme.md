# Brain Tumor Classification Using CNN 

This project implements a **Convolutional Neural Network (CNN)** to classify brain MRI scans into four categories: **Glioma, Meningioma, Pituitary Tumor, and No Tumor**. The aim is to explore how deep learning can assist in automated medical image analysis while handling real-world constraints such as limited data and class overlap.

---

## Problem Overview

Brain tumor detection from MRI scans is a complex task due to:
- High visual similarity between different tumor types  
- Variations in tumor size, shape, and contrast  
- Limited availability of labeled medical imaging data  

Manual diagnosis is time-consuming and can vary between experts. This project investigates how CNNs can learn discriminative spatial features to support automated tumor classification.

---

## Dataset

- Source: **Kaggle Brain MRI Dataset** *[Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/lakshancooray23/tumour-classification-images)*  
- Classes:
  - Glioma  
  - Meningioma  
  - Pituitary Tumor  
  - No Tumor  
- Structure:
  - `Training/` → training MRI images  
  - `Testing/` → unseen images for evaluation  

---

## Model & Training Strategy

### Architecture
- Sequential CNN with:
  - Multiple **Conv2D + MaxPooling2D** layers for hierarchical feature extraction  
  - **Batch Normalization** to stabilize learning  
  - **Dropout layers** to reduce overfitting  
  - **Softmax output layer** for multi-class classification  

### Training Process
- Optimized using the **Adam optimizer** with categorical cross-entropy loss  
- Trained over multiple epochs while monitoring validation performance  
- **EarlyStopping** was used to prevent the model from memorizing the training data  

---

## Performance & Accuracy Analysis 

- The model achieved **~81% training accuracy by the 12th epoch**, indicating effective feature learning without excessive overfitting.
- Accuracy improved gradually across epochs as the CNN learned:
  - Low-level features (edges, textures) in early epochs  
  - Higher-level tumor-specific patterns in later epochs  

### Why 81% Accuracy Is Meaningful
- The task involves **four visually similar medical classes**, making it significantly harder than binary classification  
- MRI datasets are typically **small and noisy**, which limits achievable accuracy  
- The model prioritizes **generalization** over aggressive optimization  

Rather than pushing for artificially high accuracy, the training was stopped once performance stabilized, ensuring the model remains robust on unseen data.

---

## Evaluation

To ensure transparent and reliable evaluation:
- Generated **confusion matrices** for both training and testing sets  
- Computed **precision, recall, and F1-score** for each class  

These metrics helped identify:
- Strong performance on clearly distinguishable tumor types  
- Overlap between visually similar classes such as Glioma and Meningioma  

---

## Why This Approach Works

- **CNNs** effectively capture spatial and textural patterns in MRI images  
- **Regularization techniques** (Dropout, EarlyStopping) help control overfitting  
- **Performance analysis beyond accuracy** ensures realistic model assessment  

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib  
- Scikit-learn 

---

## Notes

- Accuracy can be further improved with:
  - Larger datasets  
  - Transfer learning from pre-trained medical models  
  - Class balancing techniques
    
## Performance & Evaluation

**Model Training:**  
The CNN was trained over multiple epochs and achieved a peak training accuracy of **81% at the 12th epoch**, indicating effective learning of tumor-related features from the MRI scans.

**Final Evaluation:**  
After training was completed, the model was evaluated on a **separate Test Set** to assess its ability to generalize to unseen MRI images. This step ensures that the reported performance is not limited to the training data.

**Confusion Matrix:**  
The confusion matrices present the final classification results on the Test Set. While training accuracy reflects how well the model learned during training, the confusion matrix provides deeper insight into how accurately the model distinguishes between specific tumor types (Glioma, Meningioma, Pituitary, and No Tumor) at the end of the training process.

This project demonstrates a **realistic and defensible deep learning pipeline** for medical image classification.

---


