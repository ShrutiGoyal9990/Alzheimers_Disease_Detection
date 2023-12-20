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

## 3. Neural Networks Used

## 3.1. CNN (Convolutional Neural Network)
![model_cnn_architecture (1)](https://github.com/ShrutiGoyal9990/Alzheimers_Disease_Detection/assets/121054868/0025173d-77f2-4878-bfcd-ddec8fc02608)

CNN is an artificial neural network which consists of multiple layers that extract features through convolutional operations and learn patterns hierarchically. It uses the Sequential API in Keras, a popular deep learning framework. It begins with a rescaling layer, standardizing pixel values. Successive layers include convolutional and pooling layers, which detect patterns and downsample the data, reducing computation while preserving essential features. Dropout layers randomly deactivate neurons, preventing overfitting by enhancing generalization. The Flatten layer reshapes the output for fully connected layers, followed by dense layers that perform classification. The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss (suited for integer-encoded labels), and accuracy as the metric for evaluation.

Accuracy obtained (in %) : 99.53

## 3.2. ResNet (Residual Network)
![ResNet_img](https://github.com/ShrutiGoyal9990/Alzheimers_Disease_Detection/assets/121054868/2f042cc3-2954-4d70-b392-e7eb74a6567f)

Model Architecture : 

![model_res_architecture (1)](https://github.com/ShrutiGoyal9990/Alzheimers_Disease_Detection/assets/121054868/e45223ff-e066-463e-85b1-50330d45d68b)

ResNet (Residual Network) is a deep convolutional neural network architecture designed to tackle vanishing gradient issues in very deep networks. It introduces skip connections, allowing the network to bypass certain layers and facilitating the flow of original information alongside learned features.

This base model's layers are made non-trainable to retain pre-learned features. A new model is constructed atop the base, starting with a Global Average Pooling layer to reduce spatial dimensions and extract features. Dense layers are added for classification, followed by a softmax layer for multi-class prediction. The model is compiled using the Adam optimizer with a low learning rate for fine-tuning. The architecture leverages the ResNet50's powerful feature extraction while adapting it for a specific classification task involving four classes related to dementia severity.

Accuracy Obtained(in %) : 97.98

## 3.3. EfficientNetB0 
EfficientNetB0 Architecture : 

![EfficientNetB0_img](https://github.com/ShrutiGoyal9990/Alzheimers_Disease_Detection/assets/121054868/09518360-cb36-4832-aaeb-97a2692bb302)

Model Architecture : 

![model_eff_architecture (1)](https://github.com/ShrutiGoyal9990/Alzheimers_Disease_Detection/assets/121054868/cd4f7dd6-97e2-4fad-9bed-d8e667679213)

EfficientNetB0 belongs to the EfficientNet family, renowned for their balance of model size and performance. It initializes an EfficientNetB0 model pre-trained on ImageNet, exploiting its learned representations for transfer learning. By setting the base model's layers as trainable, it allows fine-tuning to adapt to a new classification task. A new model is constructed atop the base, starting with a Global Average Pooling layer to condense spatial information. This is followed by Dense layers for classification, concluding with a softmax layer for multi-class prediction. The model is compiled using the Adam optimizer with a moderate learning rate.

Accuracy Obtained(in %) : 97.04

## 3.4. VGG16 (Visual Geometry Group - 16)
VGG - 16 Architecture : 

![vgg_img](https://github.com/ShrutiGoyal9990/Alzheimers_Disease_Detection/assets/121054868/24764828-a9a2-4e7a-9a21-9d7af0a8349b)

Model Architecture : 

![model_vgg_architecture (1)](https://github.com/ShrutiGoyal9990/Alzheimers_Disease_Detection/assets/121054868/ce83f95e-989b-4b4f-9d1c-1a3efdf68bb5)

VGG16 is a convolutional neural network architecture known for its simplicity and effectiveness. Pre-trained on the ImageNet dataset, the provided code initializes the VGG16 model for transfer learning. The base model's layers are employed without the final classification layers, enabling it to capture intricate visual features. By setting these pre-trained layers as trainable, the model can fine-tune these learned representations to a new task.

A new model is constructed by adding layers atop the base, including a Global Average Pooling layer for spatial reduction and Dense layers for classification. The architecture culminates in a softmax layer for multi-class prediction, addressing a specific task encompassing four dementia severity classes. The model is compiled with an Adam optimizer using a modest learning rate for effective optimization.

Accuracy Obtained(in %) : 90.34

## 4. Results 
The following figure shows the accuracies of all the models used.
![accuracy_comparison](https://github.com/ShrutiGoyal9990/Alzheimers_Disease_Detection/assets/121054868/0a0ee059-da87-481d-90b8-92bdf7b3fcbd)

## 5. Conclusion
This project focuses on classifying preprocessed MRI scans of Alzheimer's patients into four distinct categories: Mild Demented, Moderate Mild Demented, Non-Mild Demented, and Very Mild Demented. It employs four different neural network architectures—CNN, ResNet, EfficientNetB0, and VGG16—to accomplish this classification task. Notably, among these architectures, the CNN framework emerges as the most effective, achieving an exceptional accuracy rate of 99.53%. This high accuracy position of the CNN model indicates its superior performance compared to other neural network frameworks in accurately identifying different stages of Alzheimer's disease based on MRI image analysis. 

As a result, this system stands as a valuable tool, aiding neurologists in their diagnosis and assessment of Alzheimer's disease severity, leveraging the precision and reliability offered by the CNN model's classification capabilities.
