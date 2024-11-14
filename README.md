# Project Overview
This project introduces a Convolutional Neural Network (CNN)-based solution for automating thyroid disease classification, leveraging ultrasound imaging and clinical data. The model enhances diagnostic accuracy by automating the classification process, making it scalable for clinical use.

# Problem Statement
Traditional thyroid disease diagnosis involves complex and time-consuming processes that are prone to human error. This project addresses these limitations by implementing an automated system using deep learning, specifically CNNs, to classify thyroid conditions effectively from ultrasound images. This approach minimizes the dependency on radiologists and is particularly beneficial for low-resource settings.

# Objectives
Assess the effectiveness of CNNs in classifying thyroid conditions using ultrasound and clinical data.
Improve diagnostic accuracy and scalability.
Implement a dynamic model that continuously learns from new data, enhancing its accuracy over time.

# Methodology
# Data Acquisition:

Collected 480 grayscale 2D thyroid ultrasound images, resized to 224x224 pixels, from Nepalgunj and Lumbini Medical Colleges in Nepal.

# Preprocessing:

Images were enhanced through data augmentation techniques like rotation, zoom, and shifts.

# Segmentation:

K-means clustering algorithm isolates the thyroid gland in each image, reducing background noise and enhancing model focus.

# Model Architecture:

A sequential CNN with multiple convolutional layers, pooling layers, dropout, and fully connected layers was developed using Keras. Softmax is used in the output layer to provide probabilistic classifications.

# Results
Achieved an accuracy rate of 86% with stable training and validation accuracy.
Model performed effectively in classifying ultrasound images and demonstrated robustness in real-world image variations.

 # Performance Analysis
Accuracy and loss metrics were monitored across 25 training epochs.
Occlusion sensitivity analysis validated the model’s focus on critical thyroid image areas, enhancing diagnostic reliability.

 # Conclusion
The CNN-based thyroid classification system successfully automates the diagnostic process, achieving high accuracy and improving over traditional methods. This approach holds significant potential for deployment in clinical settings, especially where trained radiologists are unavailable.
