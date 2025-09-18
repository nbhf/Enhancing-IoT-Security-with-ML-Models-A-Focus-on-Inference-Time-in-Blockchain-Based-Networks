# IoT Security with Machine Learning: Threat Detection

⚠️ This project is focused solely on detecting and classifying the type of cyber threat present in IoT network traffic; it does not directly counter or block the attacks themselves.

This repository contains the implementation and testing of machine learning models for **real-time IoT network security**, focusing on the critical balance between **detection accuracy** and **inference speed**. Our research addresses the growing challenge of securing IoT devices against cyber threats while ensuring security systems respond fast enough for real-time protection.

We tested **7 machine learning algorithms** on two standard cybersecurity datasets to find the optimal balance between accuracy and speed for IoT security applications:

### Algorithms Tested
- Random Forest
- Decision Tree  
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Logistic Regression
- AdaBoost
- Naïve Bayes

### Datasets Used
- **NSL-KDD**: 148,517 samples with improved data quality
- **KDD Cup-99**: 1,074,992 samples with comprehensive attack scenarios

Both datasets include 4 attack categories: DoS, Probe, R2L (Remote-to-Local), and U2R (User-to-Root)[1].

## Key Findings

### 🏆 Random Forest
**Best overall performance** combining accuracy and speed:

| Dataset | Accuracy | Inference Time |
|---------|----------|----------------|
| NSL-KDD | 99.41% | 0.000026 seconds |
| KDD Cup-99 | 99.98% | 0.000032 seconds |

TO TEST OUR MODELS YOU NEED TO DOWNLOAD THE DATASETS FIRST:

KDD CUP 99(the complete one): https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

NSL KDD: https://www.kaggle.com/datasets/hassan06/nslkdd
P.S: you need to merge KDDTrain+ and KDDTest+ and name it KDDmerged

Put the datasets and the field names.csv and the code in the same directory and then run the code to test the models

You will see the resuls in the command line and then two figures will be plotted.
