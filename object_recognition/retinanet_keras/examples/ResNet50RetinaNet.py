
# coding: utf-8

# ## Load necessary modules

import matplotlib 
matplotlib.use('Agg')
# show images inline

# automatically reload modules when they have changed

# import keras
import keras

# import keras_retinanet
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import sys

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def run_on_video(video_path, video_name, model):
    cap = cv2.VideoCapture(video_path)
    import time
    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path = "out_" + video_name + "_RetinaNet_2" + ".avi"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (1280,720)) #width, height format
    i = 0
    people_frames = []
    while(cap.isOpened()):
        has_frame, frame = cap.read()
        if not has_frame: 
            break
        # Custom code here
        i += 1
        print ("Processing frame #", i)
        sys.stdout.flush()
        processed_img, label = run_model(frame, model)
        out.write(processed_img)
        print ("Label = ", label)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Finished reading video")
    print ("Wrote video to ", out_path)
    print ("Frames with person", people_frames)
    sys.stdout.flush()
    cap.release()
    out.release()
    
def run_on_image(image_path, model):
    image = read_image_bgr(image_path)
    run_model(image, model)

def run_model(image, model):
    # copy to draw on
    draw = image.copy()
    print ("Original Frame is = ", draw.shape)
    
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    print ("Processed Frame is = ", image.shape)
    
    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    sys.stdout.flush()
    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break
        print ("Found label = ", label)
        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    plt.imshow(draw)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='') #canvas to string
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) #string to array
    data = data[:, :, ::-1]
    plt.close() 
    print("Data shape = " , data.shape)
    return data, label

def main():
    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())
    ## Load RetinaNet model
    # adjust this to point to your downloaded/trained model
    model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.0.3.h5')
    # load retinanet model
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    print("Model Loaded")
    sys.stdout.flush()
    #run_on_image('000000008021.jpg', model)
    run_on_video("uber_trimmed.mp4", "uber_trimmed", model)
    
main()

