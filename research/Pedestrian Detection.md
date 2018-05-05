# Pedestrian Detection Literature Survey

Curated list of papers on the topic of Pedestrian Detection
## Papers
1. [Pedestrian detection in video surveillance using fully convolutional YOLO neural network](https://www.researchgate.net/publication/317967088_Pedestrian_detection_in_video_surveillance_using_fully_convolutional_YOLO_neural_network) 
    * Combine detection and classification tasks using CNN.
    * YOLO CNN - Drawbacks:
        * Full connected layers that obstruct applying the CNN to images with different resolution.
        * Limits the ability to disitinguish small close human figures in groups
    * Changed network architecture overcoming above drawbacks.
    * Datasets:
        * Caltech
        * Kitty
        * Moscow city surveillance data
    * Two stages of traning:
        * Pre-trained new added convolutional layers on Caltech, Kitti
            * Didn't freeze the initial layers
        * Fine tune on ow datasets
            * Learning only for new layers
     * For both stages, 
        * SGD RMSProp
        * Batch Normalisation
        
2. [Deep convolutional neural networks for pedestrian detection](https://arxiv.org/pdf/1510.03608.pdf)
      * Extends R-CNN
3. [Adaptive Feeding: Achieving Fast and Accurate Detections by Adaptively
Combining Object Detectors](https://arxiv.org/pdf/1707.06399.pdf)  
4. [Obstacle Detection Using Local Shape Context
Descriptor on Railway Track](http://www.ijircce.com/upload/2016/february/20_Obstacle.pdf)
      * Based on Local Shape Context descriptor
      * Using edge based orientation information
5. [Night-time Pedestrian Detection Based on Temperature and HOGI Feature in
Infra-red Images ](http://ijssst.info/Vol-17/No-28/paper14.pdf)
6. [Deep Learning for Robust Road Object
Detection (Thesis)](http://publications.lib.chalmers.se/records/fulltext/249747/249747.pdf)
7. [Fastest Pedestrian Detector ](http://ijcsit.com/docs/Volume%206/vol6issue02/ijcsit20150602114.pdf)
      * HOG
8. [Obstacles and Pedestrian Detection on a Moving Vehicle](https://pdfs.semanticscholar.org/d9a6/14096dba08eae1fab8fc9d48d1c8e125e0c9.pdf)
9. [Nighttime Pedestrian Detection with a Normal Camera Using SVM Classifier](https://link.springer.com/chapter/10.1007/11427445_30)
10. [Pedestrian Detection with RCNN](http://cs229.stanford.edu/proj2015/172_report.pdf) 
11. [Caltech Pedestrian Dataset: Evaluated Algorithms](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=CF8539B8E398147CAD873FCFF6234EAF?doi=10.1.1.433.5334&rep=rep1&type=pdf)
 12. [Ten Years of Pedestrian Detection, What Have We Learned?](https://arxiv.org/pdf/1411.4304.pdf)
 13. [Word Channel Based Multiscale Pedestrian Detection Without Image Resizing and Using Only One Classifier](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Costea_Word_Channel_Based_2014_CVPR_paper.pdf)
## Github Repositories
        
1. [Caltech Pedestrain Dataset Converter for Python users](https://github.com/mitmul/caltech-pedestrian-dataset-converter)
2. [Convert Caltech annontations to Yolo compatible format](https://github.com/Jumabek/convert_caltech_annos_to_yolo)     
3. [DeepPed: Deep CNN for Pedestrian Detection (2016)](https://github.com/DenisTome/DeepPed)
4. [LSTM-RCNN Pedestrian Detection](https://github.com/buffer51/lstm-rcnn-pedestrian-detection)
5. [Segmentation Based Approach for Pedestrian Detection](https://github.com/colegulino/Deep-Neural-Networks-for-Pedestrian-Detection)
6. [Ped Dect using SVM](https://github.com/WuLC/Pedestrian-Detection)  

## All and Sundry on Object Detection in general:
1. [Deep Learning in Video Surveillance Slides](https://www.ee.cuhk.edu.hk/~xgwang/MSF.pdf)
2. [Selective Search Poster](https://www.koen.me/research/pub/vandesande-iccv2011-poster.pdf)
3. [Pedestrian Detection and Distance Estimation Slides](https://www.slideshare.net/omidAsudeh/real-time-pedestrian-detection-tracking-and-distance-estimation?qid=9fb687b8-d1b7-4094-8d5a-e185a8dbe1a8&v=&b=&from_search=1)
      * [Code](https://sourceforge.net/projects/pedestriandetectiontracking/)

