# Automated Pneumonia Detection from Chest X-Ray Using Deep Learning

This project presents an automated and interpretable deep learning system for detecting
pneumonia from chest X-ray images. The approach leverages transfer learning with a
pretrained DenseNet121 model to achieve strong classification performance while
maintaining transparency through visual explanations.

The system is designed to support reliable medical image analysis by combining robust
training strategies, comprehensive evaluation metrics, and explainable AI techniques.

---

## Overview
Pneumonia is a serious respiratory infection that requires timely and accurate diagnosis.
Chest X-ray imaging is widely used, but manual interpretation can be challenging due to
image variability, overlapping anatomical structures, and subtle disease patterns.

This project explores a deep learning–based solution for pneumonia detection using
convolutional neural networks. The focus is not only on achieving high accuracy but also
on providing interpretable predictions through Grad-CAM visualizations that highlight
clinically relevant lung regions.

---

## Key Features
- Binary classification: Pneumonia vs Normal
- Transfer learning using DenseNet121 pretrained on ImageNet
- Two-stage training strategy (feature extraction + fine-tuning)
- Data augmentation for improved generalization
- Handling of class imbalance using class weighting
- Comprehensive evaluation metrics
- Explainable AI using Grad-CAM heatmaps
- Robust and interpretable diagnostic pipeline

---

## Dataset
The project uses the Chest X-ray Pneumonia dataset available on Kaggle (originally from
the UCI Machine Learning Repository).

Dataset characteristics:
- Chest X-ray images (anterior–posterior view)
- Two classes: NORMAL and PNEUMONIA
- Predefined train, validation, and test splits
- Class imbalance with pneumonia cases as the majority class
- Real-world noise and variability

The dataset is accessed programmatically to ensure reproducibility.

---

## Data Preprocessing
To ensure data quality and reliable training, the following preprocessing steps are
applied:

- Detection and removal of corrupted or unreadable images
- Verification of image integrity, format, and dimensions
- Correct label assignment based on directory structure
- Data normalization
- Data augmentation (horizontal flipping, zooming, shearing)
- Class weighting to address class imbalance

---

## Methodology

### 1. Transfer Learning
A DenseNet121 architecture pretrained on ImageNet is used as the feature extraction
backbone. The original classification layers are removed and replaced with a custom
classification head consisting of:
- Global Average Pooling
- Fully connected layers
- Sigmoid activation for binary classification

During the initial training phase, all pretrained layers are frozen and only the custom
classification head is trained.

---

### 2. Fine-Tuning
To further improve performance, fine-tuning is applied by unfreezing the last convolutional
blocks of DenseNet121. The model is retrained using a very low learning rate to adapt
high-level features to chest X-ray–specific patterns while preserving pretrained
representations.

This two-stage training strategy improves accuracy and generalization.

---

### 3. Model Interpretability with Grad-CAM
To address the black-box nature of deep learning models, Grad-CAM (Gradient-weighted
Class Activation Mapping) is integrated.

Grad-CAM generates heatmaps that highlight regions of the chest X-ray that contribute
most to the model’s predictions. These visual explanations confirm that the model focuses
on medically relevant lung areas, enhancing transparency and trust.

---

## Evaluation Metrics
Model performance is evaluated using multiple clinically relevant metrics:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- Confusion Matrix
- AUC-ROC

Final test accuracy achieved:
- **91%**

The results demonstrate strong sensitivity in detecting pneumonia cases, minimizing
false negatives.

---

## Technologies Used
- Python
- TensorFlow / Keras
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Deep Learning
- Transfer Learning
- Explainable AI (Grad-CAM)

---

## Project Structure
pneumonia-detection/
notebooks/
  pneumonia_detection.ipynb
data/
  train/
  val/
  test/
models/
figures/
README.md

---

## How to Run

1. Clone the repository:
git clone https://github.com/USERNAME/REPO_NAME.git

2. Navigate to the project directory:
cd REPO_NAME

3. Install dependencies:
pip install tensorflow numpy pandas matplotlib opencv-python

4. Open the notebook:
jupyter notebook notebooks/pneumonia_detection.ipynb

5. Run all cells in order.

---

## Results Summary
- Accurate and robust pneumonia classification
- Effective use of transfer learning
- Improved performance through fine-tuning
- Visual interpretability with Grad-CAM
- Clinically meaningful decision support

---

## Future Improvements
- Training on larger and more diverse datasets
- Multi-class classification (bacterial vs viral pneumonia)
- Integration with clinical metadata
- Deployment as a web or desktop application
- Further validation on external datasets

---

## License
This project is provided for research and personal use.
