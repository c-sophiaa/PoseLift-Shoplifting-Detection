# PoseLift: Shoplifting Detection

A computer vision and machine learning project that explores shoplifting detection by analyzing human pose sequences in surveillance video.

## Overview

This project investigates whether human skeletal movement patterns can be used to distinguish between normal and shoplifting behavior. The pipeline uses YOLOv8 Pose to extract human keypoints from video frames and a TensorFlow GRU neural network to classify temporal pose sequences.

## Tech Stack

- Python
- YOLOv8 Pose
- OpenCV
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Jupyter Notebook

## Pipeline

1. Process surveillance video using OpenCV.
2. Detect people and extract skeletal keypoints using YOLOv8 Pose.
3. Convert pose information into temporal sequences.
4. Prepare and pad sequential data for model training.
5. Train a TensorFlow GRU neural network for binary classification.
6. Evaluate predictions as normal or shoplifting behavior.

## Model Evaluation

Model performance was evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1-score
- Confusion matrices

The experiments showed limited classification performance, with approximately 57% test accuracy in the final evaluation. This project therefore serves as an exploration of pose-based action classification rather than a production-ready shoplifting detection system.

## Repository Structure

```text
PoseLift-Shoplifting-Detection/
├── notebooks/
│   ├── pose_extraction.ipynb
│   └── gru_training.ipynb
├── .gitignore
└── README.md