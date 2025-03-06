# Facial Emotion Recognition (FER) Using the FER13 dataset

## Introduction
In this project, I focused on developing a Facial Emotion Detection System using deep learning techniques. The objective is to detect human emotions from facial expressions in real-time and use these detected emotions to personalize content for users on platforms like social media.

## Objective
The key objective of this project is:

**Real-Time Emotion Analysis:** Analyze users' emotions based on their facial expressions in images shared on social media, providing real-time emotion detection capabilities.

**Content Personalization:** Use the detected emotional state to dynamically personalize content shown to the user, enhancing their engagement by recommending posts, videos, or articles that match their current mood.

## Use Cases
I designed this project with several real-world use cases in mind:

**Social Media Monitoring:** Detect the emotional trends of users in real-time and identify sentiment shifts across platforms.
**Personalized Recommendations:** Suggest content (like videos, articles, or ads) that aligns with the detected mood, improving user retention and engagement.
**Mental Health Tracking:** By detecting recurring negative emotions, platforms could provide supportive content or even suggest professional help if needed.
**Brand Sentiment Analysis:** Companies could monitor how users emotionally react to brand posts or advertisements using facial expression analysis.

## Dataset Description
I worked with the FER-2013Plus dataset, which is a widely used benchmark for facial emotion recognition tasks.

**Number of images:** 35,485

**Image size:** 48x48 pixels (grayscale)

**Emotions covered:**

Anger
Contempt
Disgust
Fear
Happiness
Neutral
Sadness
Surprise

![image](https://github.com/user-attachments/assets/7380aaaf-fef8-4a21-9151-e173f9dbdd69)

## **Train set:** 28,386 images

![image](https://github.com/user-attachments/assets/27889cb9-15dd-48f3-82e8-ee921bdeed0c)

## **Test set:** 7,099 images

![image](https://github.com/user-attachments/assets/69cf75e0-9f5d-4ff1-92a5-9754ec84c325)

## Preprocessing
Since the images were low resolution, I focused heavily on preprocessing to make sure the data was ready for training. The steps I applied included:

Resizing all images to 48x48 pixels to maintain uniformity.
Reshaping images to (48, 48, 3) to match ResNet50 input shape
Converting all images to grayscale to reduce complexity.
Normalization by scaling pixel values to [0,1].
Applying data augmentation to improve generalization. This included:
Random rotations
Horizontal flips
Random zooming
Width and height shifts

![image](https://github.com/user-attachments/assets/21c0343e-97e6-45e6-a1c0-b2ff3609b75b)


![image](https://github.com/user-attachments/assets/aced3071-9a3d-45e3-95fa-33b3d671184d)

## Model Development
I experimented with two different models:

## 1. Custom CNN
I designed a Convolutional Neural Network from scratch, specifically keeping the small grayscale images in mind. The CNN consists of:

Convolutional layers to extract spatial features.
MaxPooling layers to downsample feature maps.
Fully connected dense layers to capture high-level patterns.
Dropout layer to reduce overfitting.
Final softmax layer with 8 output neurons for the 8 emotions.

![image](https://github.com/user-attachments/assets/905a3bd0-9c46-41b6-a9ad-2a7d51d5753c)

 Total params: 25,694,088 (98.02 MB)
 Trainable params: 2,106,376 (8.04 MB)
 Non-trainable params: 23,587,712 (89.98 MB)

 ![image](https://github.com/user-attachments/assets/4f400236-8e12-485d-b7f3-8a07c7f775cb)

![image](https://github.com/user-attachments/assets/878bbf6e-93a9-48ff-9e0a-a81ad07988c5)

## Model Accuracy and Model Loss 
![image](https://github.com/user-attachments/assets/135a4b3b-ae90-41ac-b107-0449bb79a1d8)


## 2. Transfer Learning with ResNet50
To compare performance, I also applied ResNet50, a pre-trained model that is popular for image classification. Since ResNet50 was trained on ImageNet (color images, higher resolution), I modified the input layer and added:
![image](https://github.com/user-attachments/assets/4f8b0d9a-92c2-41e7-8de6-22c946491af2)
Total params: 4,232,456 (16.15 MB)
 Trainable params: 4,232,456 (16.15 MB)
 Non-trainable params: 0 (0.00 B)
 
GlobalAveragePooling2D to flatten feature maps.
Dense layer with 1024 neurons.
Final softmax layer with 8 output neurons.
For this model, I froze most layers to preserve pre-trained features and only fine-tuned the top layers.

![image](https://github.com/user-attachments/assets/3f536a08-d9b5-49a7-b622-61a06aeaca94)

## Model Accuracy and Model Loss 

![image](https://github.com/user-attachments/assets/877355ec-a707-44a5-941c-1391ec8480bf)

![image](https://github.com/user-attachments/assets/16d601f1-978f-402f-a3b0-c91c38313db7)


## Display Prediction 
![image](https://github.com/user-attachments/assets/c30a7043-f0fb-45fe-a27f-cfbc6d6eb93d)


## What I Observed- 
### After running both the models, I observed:
The Custom CNN performed significantly better than ResNet50. This was expected because the custom CNN was designed specifically for this dataset (grayscale, low-resolution images), while ResNet50 was originally trained on high-resolution color images.
ResNet50 overfitted quickly, even with augmentation, because it was too large for the small dataset.
The custom CNN generalized better due to its simpler architecture, which fit the dataset better.
Class imbalance (some emotions having far fewer images) also affected overall accuracy.

## Model Results Comparison
![image](https://github.com/user-attachments/assets/9600faa3-3524-4f80-a5a6-67d99adb6dcf)



## Technologies Used
Since this was a deep learning project, I worked with:

## TensorFlow
Framework for building and training the CNN and ResNet50 models.
Enabled GPU acceleration for faster training.

## Keras
High-level API that made it easier to define models, layers, and training routines.
Pre-built applications like ResNet50 helped me quickly apply transfer learning.

## Scikit-learn (sklearn)
Used mainly for evaluation metrics like the classification report and confusion matrix.
Also used for basic preprocessing when needed (like label encoding).

## Suggested Application
The emotion detection pipeline I built can be integrated into larger applications, such as:
•	Emotion-based content recommendation: Suggest movies, music, or articles based on detected emotion.

•	Customer sentiment analysis: Analyze customer emotions during video calls for better support experiences.

•	Mental health monitoring: Track emotional patterns over time for early intervention.

•	Smart devices: Build smart mirrors or smart TVs that react to the user's mood.

## Key Challenges Faced
•	Working with low-resolution images (48x48 pixels) made it hard to capture fine facial details.

•	Class imbalance — some emotions (like contempt) had very few samples, while others (like neutral and happiness) dominated the dataset.

•	Pre-trained models like ResNet50 were not ideal for this dataset, which reinforced the importance of matching model architecture to data type.

•	Balancing accuracy and overfitting required careful tuning, especially with the custom CNN.

## What I Would Improve If I Do This Again
•	Explore lighter models like MobileNet that are designed for mobile and low-resolution inputs.

•	Collect higher-resolution images to capture richer facial expressions.

•	Use class balancing techniques (like oversampling minority classes) to improve generalization.

•	Extend this into a multi-modal system where facial emotion is combined with speech tone analysis and text sentiment analysis (in case captions or user comments are available).


## Conclusion
This project helped me understand the importance of designing models that match the data. While transfer learning is powerful, it does not always work well if the target data is very different from the source data the model was trained on.

By building a custom CNN, I was able to achieve much better performance for this specific task, reinforcing the importance of dataset-specific model design.

This project also gave me a clearer understanding of how facial expression analysis can power real-world applications — from personalized content delivery to real-time emotion-aware systems.

