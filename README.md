# ALZHEIMERS DISEASE DETECTION
## 1. Introduction
Alzheimer's disease (AD) is the most common form of dementia that progressively damages brain cells. It is characterized by abnormal protein deposits in the brain, Alzheimer's gradually worsens over time, affecting communication and reasoning, resulting in memory and thinking deficits, loss of basic abilities, and ultimately, death. 

Currently, the annual cost of treating AD is 1 Trillion USD and it is expected that by 2050, 152 Million peope will be affected by AD. While there is no cure for AD, its onset diagnosis can prevent it from becoming too severe thus improving patient’s life. Recent advancements in computer vision have found to be impactful, however the datasets for AD are limited and heavily imbalanced. Due to this severe class imbalance the classifiers are prone to be biased towards the majority class i.e., classifying a person with early symptoms as "Not Impaired" (No Alzheimer's) which is highly undesirable.

## 2. About the Dataset
DATASET LINK : https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset

The Data is collected from Kaggle. which  consists of Preprocessed MRI (Magnetic Resonance Imaging) Images. All the images are resized into 128 x 128 pixels. The Dataset has four classes of images.
The Dataset is consists of total 6400 MRI images.

Class - 1: Mild Demented (896 images)

Class - 2: Moderate Demented (64 images)

Class - 3: Non Demented (3200 images)

Class - 4: Very Mild Demented (2240 images)

## 3. Neural Network Used

## 3.1. CNN (Convolutional Neural Network)
![model_cnn_architecture (1)](https://github.com/ShrutiGoyal9990/Alzheimers_Disease_Detection/assets/121054868/0025173d-77f2-4878-bfcd-ddec8fc02608)

CNN is an artificial neural network which consists of multiple layers that extract features through convolutional operations and learn patterns hierarchically. It uses the Sequential API in Keras, a popular deep learning framework. It begins with a rescaling layer, standardizing pixel values. Successive layers include convolutional and pooling layers, which detect patterns and downsample the data, reducing computation while preserving essential features. Dropout layers randomly deactivate neurons, preventing overfitting by enhancing generalization. The Flatten layer reshapes the output for fully connected layers, followed by dense layers that perform classification. The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss (suited for integer-encoded labels), and accuracy as the metric for evaluation.

## 4. Results 
Accuracy obtained (in %) : 99.53

## 5. Conclusion
This project focuses on classifying preprocessed MRI scans of Alzheimer's patients into four distinct categories: Mild Demented, Moderate Mild Demented, Non-Mild Demented, and Very Mild Demented. It employs four different neural network architectures—CNN—to accomplish this classification task. The CNN framework is achieving an exceptional accuracy rate of 99.53%. This high accuracy position of the model indicates its superior performance in accurately identifying different stages of Alzheimer's disease based on MRI image analysis. 

As a result, this system stands as a valuable tool, aiding neurologists in their diagnosis and assessment of Alzheimer's disease severity, leveraging the precision and reliability offered by the CNN model's classification capabilities.
