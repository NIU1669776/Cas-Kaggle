# Cas Kaggle: ASL Sign Language Recognition
This repository contains two different Jupyter Notebooks containing the two main steps of our project. Below is a brief overview of each project.

---

## 1. Bag of Visual Words

### Description
This notebook uses the **Bag of Visual Words** technique for image classification. The algorithm generated keypoints and descriptors from images, clustering them using Kmeans to extract the most important features and using these for classification using Support Vector Machines (SVM).

### Key Steps
- Download the Dataset: https://www.kaggle.com/datasets/ayuraj/asl-dataset
- Create masks to segment the hand region.
- Convert images to grayscale and generate dense keypoints and descriptors using the previously created mask.
- Use k-means clustering to create a visual codebook.
- Train and evaluate an SVM classifier with hyperparameter tuning.

Using the predict_image function, you can predict different images, there are some examples at the end of the notebook.

NOTE: The paths used wont work unless you create the needed folders. Please, change the paths for valid ones if you plan on executing them.

### Dependencies
- `numpy`
- `opencv-python` (`cv2`)
- `matplotlib`
- `pandas`
- `scipy`
- `sklearn`
- `seaborn`
- `Pillow` (`PIL`)

---

## 2. Hand Detection

### Description
This notebook focuses on detecting and cropping hand regions in images using **MediaPipe**. The hand landmarks are identified, and bounding boxes are created to crop the hand from the background, transforming an original image into one similar to the ones used in the training dataset.\\

There is also one chapter at the end of the notebook for frame extraction. This allow us to divide a video into frames to process them after.

### Key Steps
- Load images and preprocess them for hand detection.
- Use MediaPipe for identifying hand landmarks and extracting bounding boxes.
- Apply this for any desired image or extract frames from a desired video to process them after.

### Dependencies
- `numpy`
- `opencv-python` (`cv2`)
- `matplotlib`
- `mediapipe`
- `scipy`

---
