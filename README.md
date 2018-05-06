# Enabling Pedestrian Safety through Computer Vision techniques. A case study of the 2018 Uber autonomous car crash.

## Introduction

In this work we aim to study the application of Computer Vision techniques for the purpose of detecting pedestrians in both broad-light and low-light scenarios for use in autonomous vehicles. We explore various traditional image processing methods, machine learning techniques and state-of-the-art neural network based approaches. We also identify the most suitable combination of image processing and neural network techniques that enable the highest pedestrian safety.

For this work, we use the dash-cam footage released by Uber Inc. from it's self driving car crash that happened in March 20, 2018 - Tempa, Arizona, unfortunately resulting in the death of a pedestrian. Uber Inc. claims their technology was unable to detect this pedestrian. This video has very low lighting and thus is an excellent opportunity to try a variety of algorithms and techniques in the hopes of detecting the pedestrian as early as possible.

## Repository Structure

The repository structure is as follows - 

#### **Code**
  * **class_files:** Dictionary of class mappings
  * **image_classification:** Implementation of various image classification neural networks
  * **image_processing:** Implementation of various image enhancement techniques
  * **object_recognition:** Implemenation of various Object Recognition techniques - traditional and neural network based
  * **utilities:** Helpful utility scripts
#### **Images**
  * **Frames:** Contains the original frames from the uber crash video
  * **Misc:** As the name suggests, miscalaneous images related to the project
#### **Research**
  * Contains our research notes while learning various topics, background study, prior work study, etc
#### **Videos**  
  * **Originals:** Contains the original videos on which we tested our methods against
  * **Processed:** Contains videos after applying various processing and enhancement techniques
  * **Results:** Contains the processed images after either classification or object recognition has been performed

## Image Processing/Enhancement

We applied a plethora of processing and enhancement techniques which have shown promising results. We broadly classify these into 2 categories
1. General Image Processing Techniques
2. Low-light Image Enhancement Techniques

### General Image Processing Techniques
   
* Binary Thresholding
* CLAHE (Contrast Limited Adaptive Histogram Equalization)
* Gamma Correction
* Motion Mapping
* Canny Edge Detection
   
### Low-light Image Enhancement Techniques
   
* Adaptive Thresholding + Filtering 
* Exposure Fusion Framework (Ying et al)
* Camera Response Model (Ying et al)
* LL-Net (Lowlight-net)

## Object Recognition
  
* You Only Look Once (YOLO)
* Single Shot Detector (SSD)
* RetinaNet
* Support Vector Machine with Histogram of Oriented Gradient features (SVM+HOG)
    
