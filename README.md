# Uterine Fibroid Classification Using a Convolutional Neural Network (CNN)

This project presents a deep learning approach to classify uterine fibroids from ultrasound images using a Convolutional Neural Network (CNN). The model achieves high accuracy and AUC-ROC, demonstrating the potential of automated fibroid screening in clinical workflows.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [CNN Architecture](#cnn-architecture)
- [Hyperparameters](#hyperparameters)
- [Training & Results](#training--results)
- [Interpretation](#interpretation)
- [Future Work](#future-work)
- [Conclusion](#conclusion)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Introduction
Uterine fibroids, also called leiomyomas, are benign smooth muscle tumors of the uterus that affect women of reproductive age. While often asymptomatic, fibroids may cause pelvic pain, abnormal bleeding, infertility, and pregnancy complications.

Ultrasound imaging is the standard diagnostic tool but interpretation is highly operator-dependent, which can lead to variability in diagnosis. Convolutional Neural Networks (CNNs) are effective in medical image analysis because they automatically learn hierarchical features such as textures and shapes.

This project develops a CNN model to classify uterine fibroids from ultrasound images, aiming to improve diagnostic accuracy, reduce human error, and provide a scalable automated solution for clinical workflows.

---

## Dataset
- **Source:** [Uterine Fibroid Ultrasound Images (Kaggle)](https://www.kaggle.com/)
- **Classes:** Fibroid, Non-Fibroid
- **Preparation:** Images were organized into a standard deep learning directory structure.
- **Loading:** TensorFlow’s `image_dataset_from_directory()` function was used to handle batching, resizing, shuffling, and prefetching for optimized performance and reproducibility.

---

## CNN Architecture
- Rescaling for normalization
- Convolutional layers with 32 → 64 → 128 filters for hierarchical feature extraction
- MaxPooling layers for noise reduction
- Dense layers for final classification

This architecture allows the network to capture complex patterns in ultrasound images while reducing overfitting and computational cost.

---

## Hyperparameters
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Batch Size:** 32
- **Epochs:** 10
- Progressive filter depth to strengthen feature extraction

---

## Training & Results
- **Data Split:** 80% training, 20% testing (seed=123)
- **Performance:**  
  - Test Accuracy: 96%  
  - AUC-ROC: 0.97

---

## Interpretation
The CNN effectively captured fibroid patterns from ultrasound images, demonstrating strong generalization. Running the model on distributed virtual machines mimics potential real-world hospital AI deployments.

---

## Future Work
- Advanced data augmentation to improve robustness
- Transfer learning with architectures like EfficientNet or ResNet
- Grad-CAM for model interpretability
- Real-time deployment for clinical use

---

## Conclusion
The CNN-based model achieves high accuracy and AUC-ROC for uterine fibroid classification. Proper data handling, hierarchical feature extraction, and distributed execution contribute to its performance and scalability. This work demonstrates the potential of automated ultrasound analysis as a non-invasive, reliable tool for early detection and screening of uterine fibroids.

---

## Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib
- Jupyter Notebook (optional)

---

## Usage
1. Clone the repository:
```bash
git clone https://github.com/BridgetNyamekye/uterine-fibroid-cnn.git
cd uterine-fibroid-cnn
