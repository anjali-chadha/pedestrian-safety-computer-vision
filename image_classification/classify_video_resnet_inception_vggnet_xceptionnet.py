#import the necessary packages
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2
import sys 


# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception, # TensorFlow ONLY
    "resnet": ResNet50
}

WEIGHT_PATHS = {
    "resnet": "/scratch/group/puneet-anjali-group/uber-workspace/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
    "vgg16": "/scratch/group/puneet-anjali-group/uber-workspace/vgg16_weights_tf_dim_ordering_tf_kernels.h5",
    "vgg19": "/scratch/group/puneet-anjali-group/uber-workspace/vgg19_weights_tf_dim_ordering_tf_kernels.h5",
    "inception": "/scratch/group/puneet-anjali-group/uber-workspace/inception_v3_weights_tf_dim_ordering_tf_kernels.h5",
    "xception": "/scratch/group/puneet-anjali-group/uber-workspace/xception_weights_tf_dim_ordering_tf_kernels.h5"
}

inputShape = (224, 224)

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False,
        help="path to the input image")
    ap.add_argument("-v", "--video", required=False,
        help="path to the input video")
    args = vars(ap.parse_args())

    for model in MODELS.keys():
        args["model"] = model
        if args["image"] is not None:
            image = load_img(args["image"], target_size=inputShape)
            image = img_to_array(image)
            run_model(image, image, args["model"])
        if args["video"] is not None:
            handle_video(args)
            
def handle_video(args):
    model = load_model(args["model"])
    cap = cv2.VideoCapture(args["video"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_dir  = "output/" + str(model.name)
    out_path = out_dir + "/out_" + args["video"].split(".")[0] + "_" + args["model"] + ".avi"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (1280,720))
    i = 0
    while(cap.isOpened()):
        has_frame, frame = cap.read()
        if not has_frame: 
            break
        # Custom code here 
        i += 1
        print ("Processing frame #", i)
        resize = cv2.resize(frame, inputShape) 
        processed_frame = run_model(resize.astype(float), frame, model)
        out.write(processed_frame)  
    print("Finished reading video")
    print ("Wrote video to ", out_path)
    cap.release()
    out.release()
    
def load_model(model_name):
    # load our the network weights from disk (NOTE: if this is the
    # first time you are running this script for a given network, the
    # weights will need to be downloaded first -- depending on which
    # network you are using, the weights can be 90-575MB, so be
    # patient; the weights will be cached and subsequent runs of this
    # script will be *much* faster)
    print("[INFO] loading {}...".format(model_name))
    Network = MODELS[model_name]
    model = Network(weights=WEIGHT_PATHS[model_name])
    return model

def run_model(image, original, model):
    # initialize the input image shape (224x224 pixels) along with
    # the pre-processing function (this might need to be changed
    # based on which model we use to classify our image)
    inputShape = (224, 224)
    preprocess = imagenet_utils._preprocess_numpy_input

    # if we are using the InceptionV3 or Xception networks, then we
    # need to set the input shape to (299x299) [rather than (224x224)]
    # and use a different image processing function
    if model in ("inception", "xception"):
        inputShape = (299, 299)
        preprocess = preprocess_input

    # load the input image using the Keras helper utility while ensuring
    # the image is resized to `inputShape`, the required input dimensions
    # for the ImageNet pre-trained network
    print("[INFO] loading and pre-processing image...")
    sys.stdout.flush()

    # our input image is now represented as a NumPy array of shape
    # (inputShape[0], inputShape[1], 3) however we need to expand the
    # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
    # so we can pass it through thenetwork
    image = np.expand_dims(image, axis=0)

    # pre-process the image using the appropriate function based on the
    # model that has been loaded (i.e., mean subtraction, scaling, etc.)
    image = preprocess(image, None, "caffe")

    # classify the image
    print("[INFO] classifying image with '{}'...".format(model.name))
    sys.stdout.flush()
    preds = model.predict(image)
    print("predictions are = ")
    print(preds.shape)
    sys.stdout.flush()
    P = imagenet_utils.decode_predictions(preds)

    # loop over the predictions and display the rank-5 predictions +
    # probabilities to our terminal
    for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
        sys.stdout.flush()
    # load the image via OpenCV, draw the top prediction on the image,
    # and display the image to our screen
    #orig = cv2.imread(image)
    (imagenetID, label, prob) = P[0][0]
    cv2.putText(original, "Label: {}, {:.2f}%".format(label, prob * 100),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    sys.stdout.flush()
    return original
                
main()
