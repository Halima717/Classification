# Bear vs Panda Image Classification 

This project demonstrates **image classification** using **transfer learning** with the **VGG16 model** in **TensorFlow/Keras** to classify images of **bears and pandas**.

## Dataset 

- **Source:** [Pandas vs Bears Kaggle Dataset](https://www.kaggle.com/datasets/mattop/panda-or-bear-image-classification?select=PandasBears)  
- **Structure:**  
  - `Train/` → training images  
  - `Test/` → validation images  

## Approach 

### Data Preprocessing & Augmentation
- Rescaled images to `[0,1]`  
- Applied random **rotations**, **flips**, **zooms**, **shifts**, and **shears** to increase dataset variability  

### Model Architecture
- Used **VGG16** pre-trained on ImageNet as the base  
- Added custom layers:  
- Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)
- Used **binary cross-entropy** as the loss function and **Adam optimizer**  

### Training
- Trained for **10 epochs** with **early stopping**  
- Fine-tuned the last few layers of VGG16 with a **lower learning rate**  

## Evaluation 
- Achieved high accuracy on both training and validation sets (note: the dataset is small, so overfitting is possible)  
- Model can predict individual images as **bear** or **panda**  


## Usage 

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = load_model('vgg16_bear_panda.h5')

# Load and preprocess an image
img_path = 'path_to_image.png'
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction_prob = model.predict(img_array)
predicted_class = (prediction_prob > 0.5).astype(int)
class_names = ['bear', 'panda']
predicted_class_name = class_names[predicted_class[0][0]]

print(f"Predicted class: {predicted_class_name}")

plt.imshow(img)
plt.title(f"Predicted: {predicted_class_name}")
plt.axis('off')
plt.show()
```
## Notes 

- The model may overfit due to the small dataset size.
- Data augmentation helps, but more images would improve generalization.
