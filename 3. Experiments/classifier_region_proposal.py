"""
    Author: Alexander Shellabear
    Email: alexshellabear@gmail.com

    Research
    1) Step by step explanation
        https://dongjk.github.io/code/object+detection/keras/2018/05/21/Faster_R-CNN_step_by_step,_Part_I.html
    2) vgg with top=false will only output the feature maps which is (7,7,512), other solutions will have different features produced
        https://github.com/keras-team/keras/issues/4465
    3) Understanding anchor boxes
        https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/
    4) Faster RCNN - how they calculate stride
        https://stats.stackexchange.com/questions/314823/how-is-the-stride-calculated-in-the-faster-rcnn-paper
    5) Good article on Faster RCNN explained, can only access via print to pdf
        https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8
    6) Indicating that Anchor boxes should be determine by ratio and scale
        ratio should be width:height of 1:2 1:1 2:1
        scale should be 1 1/2 1/3
        https://keras.io/examples/vision/retinanet/
    7) Best explanation of anchor boxes
        https://www.mathworks.com/help/vision/ug/anchor-boxes-for-object-detection.html#:~:text=Anchor%20boxes%20are%20a%20set,sizes%20in%20your%20training%20datasets.
    8) Summary of object detection history, interesting read
        https://dudeperf3ct.github.io/object/detection/2019/01/07/Mystery-of-Object-Detection/
    9) What to do if there are no good potential bounding boxes have a high enough IOU.
       Suggestion is to just get the next best option.
       https://github.com/you359/Keras-FasterRCNN/blob/eb67ad5d946581344f614faa1e3ee7902f429ce3/keras_frcnn/data_generators.py#L203 
    10) Exactly what I was wanting on RPN - have to use pdf
        https://towardsdatascience.com/building-a-mask-r-cnn-from-scratch-in-tensorflow-and-keras-c49c72acc272
    11) Custom loss functions in Keras
        https://stackoverflow.com/questions/54977311/what-is-loss-cls-and-loss-bbox-and-why-are-they-always-zero-in-training
    12) Line 5, tells what the bounding box format should be before passing it to the neural net
        Must be a numpy array of [x_centre,y_centre,width,height]
        Normalise all values using a np.std
        https://github.com/jrieke/shape-detection/blob/master/single-rectangle.ipynb

    Lessons Learnt
        1) Do not exceed the length of the image with the anchor boxes
        2) np.array.reshape() allows you to change the change of your array, in this case you can flatten your array using
            np.array.reshape(-1), this will let np choose the shape as long as it all adds up to np.array.shape
            np.array.reshape(-1,1) means the array will be flattened but each value will have it's own cell
            https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
        3) Array slicing with numpy or lists
            list[start:stop:step]
            list[:] = all
            list[0::2] = start at 0 and only return every 2nd item
        4) If you want to explicitely add a new axis with the list then you can use
            np.newaxis which explicitely adds a new axis
            only works for numpy arrays not lists
            https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it
        5) using bounding box regression means that the output of the RPN is in the following
            deltas (slight movements) = [dx_point_c,dy_point_c,dw_in_log_space,dh_in_log_space]
            It is interesting that regression for width and height are in the log space
            https://datascience.stackexchange.com/questions/30557/how-does-the-bounding-box-regressor-work-in-fast-r-cnn
        6) Originally was really confused as to why the output of the Region Proposal Network RPN was not giving the number
            of rows output was different to the number of proposed bounding boxes. I accidentally passed the number of anchor
            boxes per anchor aka the number of potential boxes per anchor point.
            I was only passing the dimensions of all potential boxes per anchor point
            This is also including those boxes which fall outside of the image
        7) If something is too difficult to understand make the problem simple. Break it into smaller sub problems that can be 
            solved and then added together.
        8) To get the first data point out of a generator in python use the next function. For example...
            Getting the child files of a path next(os.walk(path))[1]  
        9) Use the .get(item_name,"default if there is no item_name key") function to get a key from the specified item_name. 
            If it does not exist instead of erroring out you can pass it a default function
        10) The input size does not need to be 224,224 when using a vgg16 model that is toppless.
            This will change the size of the output feature map
        11) You can have a custom loss function and still have a model checkpoint.
            Importantly you can check out what metrics your model has through print(model.metrics_names)
            Then just add val to the front
            https://stackoverflow.com/questions/43782409/how-to-use-modelcheckpoint-with-custom-metrics-in-keras 
        12) Can't use image generator and flip the feature map data because it is not representing an image but instead is representing features
        13) Easiest way to access a pixel is flipped from x y to y x in cv2 and np
            https://stackoverflow.com/questions/54549322/why-should-i-use-y-x-instead-of-x-y-to-access-a-pixel-in-opencv
        14) To increase the image size by adding images to the left and right of each other follow this post
            https://stackoverflow.com/questions/7589012/combining-two-images-with-opencv
            To stack vertically (img1 over img2): vis = np.concatenate((img1, img2), axis=0)
            To stack horizontally (img1 to the left of img2): vis = np.concatenate((img1, img2), axis=1)

    Goal
        Create a simple classifier from VGG feature map
        1) Take 2 videos of object with and without
        2) Extract data
        3) Create conv2d with binary classifier, 
        4) Train 
        5) Check results
        6) Publish to own git hub repo
        7) Tidy up & send to Cheng

"""

import cv2
import numpy as np
from keras import applications
from keras import layers
from keras import Model
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import datetime
import pickle
import sys # Used to ensure packages are installed correctly
import os
try:
    from ml_library import dataset_util
    from ml_library.utility_functions import *
except:
    print("Hrmm the import failed, likely you'll need to move the 'ml_library' folder to the same directory as this file")
    exit()


def create_dataset_for_training(human_readable_dataset):
    """
        produces a dataset in a format which ML can train on 
        Assumption 1: The human_readable_dataset will be a list of dictionaries containing
            FeatureMap: The output from the vgg16 classifier model
            GroundTruthOutput: The ground truth represented in the output array from the half assed RPN

    """
    x_data = np.array([row["FeatureMap"][0] for row in human_readable_dataset])
    y_data = np.array([row["GroundTruthOutput"][0] for row in human_readable_dataset])
    return x_data, y_data

def create_todays_date_for_callback(folder:str, name_of_file:str):
    """
        Creates the name for training data to be saved. Has a sub folder with the start date of training and model names are put there. 
    """
    todays_date_string = datetime.datetime.today().strftime("%d %B %Y %Hhrs")
    if not os.path.exists(f"{folder}{os.sep}{todays_date_string}"):
        os.mkdir(f"{folder}{os.sep}{todays_date_string}")
    file_path = f"{folder}{os.sep}{todays_date_string}{os.sep}{name_of_file}"
    return file_path


def create_classifier(classifier_channels):
    """
        Creates the RPN classifier model
    """
    assert type(classifier_channels) == int # Should be an integer

    # Input layer, should have the same number of channels that the headless classifier produces
    feature_map_input = layers.Input(shape=(None,None,classifier_channels),name="RPN_Input_Same")
    # CNN component, ensure that padding is the same so that it has the same dimensions as input feature map
    convolution_3x3 = layers.Conv2D(filters=classifier_channels,kernel_size=(3, 3),name="3x3",padding="same")(feature_map_input)
    # Output objectivness
    objectivness_output_scores = layers.Conv2D(filters=1, kernel_size=(1, 1),activation="sigmoid",kernel_initializer="uniform",name="scores1")(convolution_3x3)
    # Create model with input feature map and output
    model = Model(inputs=[feature_map_input], outputs=[objectivness_output_scores])
    # Set loss and compile
    model.compile(optimizer='adam', loss={'scores1':losses.binary_crossentropy})

    return model

if __name__ == "__main__":
    print("starting...")

    # get dataset, TODO convert to function later TODO make into vgg
    config = { 
        "ObjectVideoPath" : r"3. Experiments\1. Binary Classifier Data\object.mp4"
        ,"NoObjectVideoPath" : r"3. Experiments\1. Binary Classifier Data\no_object.mp4"
        ,"ProcessedObjectFolder" : r"3. Experiments\1. Binary Classifier Data\object_processed"
        ,"ProcessedNoObjectFolder" : r"3. Experiments\1. Binary Classifier Data\no_object_processed"
        ,"SaveFinalDataset" : r"3. Experiments\1. Binary Classifier Data\data_set.p"
        ,"ModelsDirectory" : r"3. Experiments\2. Models"
        ,"BaseNameModelFile" : "basic_rpn"
        ,"NumberOfFramesToGet" : 150
        ,"ScreenSize" : (500,750)
        ,"ModelInputSize" : (224,224)
    }

    # Get vgg model without top to extract features
    vggmodel = applications.VGG16(include_top=False,weights='imagenet')
    config["VGG16Model"] = vggmodel
    
    dataset = dataset_util.dataset_generator(config)
    dataset.convert_video_to_data(config['ObjectVideoPath'])
    """
    if os.path.exists(config["SaveFinalDataset"]):
        human_readable_dataset = pickle.load(open(config["SaveFinalDataset"],"rb"))
    else:
        human_readable_dataset = {}
        human_readable_dataset["Objects"] = feature_map_each_frame_in_video_and_assign_label(config["ObjectVideoPath"],vggmodel,"Object",max_frames=500)
        #human_readable_dataset["NoObjects"] = feature_map_each_frame_in_video_and_assign_label(config["NoObjectVideoPath"],vggmodel,"No Object",max_frames=100)
        pickle.dump(human_readable_dataset,open(config["SaveFinalDataset"],"wb"))
    """
    x_data, y_data = create_dataset_for_training(human_readable_dataset["Objects"])
    x_train, x_test , y_train, y_test = train_test_split(x_data,y_data,test_size=0.10)

    # Can't use image generator
    # trdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
    # traindata = trdata.flow(x=x_train, y=y_train)
    # tsdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
    # testdata = tsdata.flow(x=x_test, y=y_test)

    model = create_classifier(512)

    check_point = ModelCheckpoint(filepath=create_todays_date_for_callback(config["ModelsDirectory"], config["BaseNameModelFile"])+"-{epoch:02d}-{val_loss:.6f}.hdf5",
                                              monitor="val_loss",
                                              mode="min",
                                              save_best_only=True,
                                              )

    model.fit(x=x_train,y=y_train, batch_size=8, epochs=30, verbose=1,validation_split=0.1,callbacks=[check_point])
    
    index_to_select = 343

    img = human_readable_dataset["Objects"][index_to_select]["InputImage"]
    mask = human_readable_dataset["Objects"][index_to_select]["ObjectMask"]
    featuremap = human_readable_dataset["Objects"][index_to_select]["FeatureMap"]
    ground_truth = human_readable_dataset["Objects"][index_to_select]["GroundTruthOutput"][0]
    prediction = model.predict(featuremap)[0] 

    cv2.imshow("Input",img)
    cv2.imshow("mask",mask)
    cv2.imshow("OutputPrediction", cv2.resize(prediction,(224,224)))
    cv2.imshow("ground_truth", cv2.resize(ground_truth,(224,224)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # feature_to_input_x_scale, feature_to_input_y_scale, feature_to_input_x_offset, feature_to_input_y_offset = find_feature_map_to_input_scale_and_offset(single_input_image,single_feature_map[0])
    # coordinate_of_anchor_boxes = get_coordinates_of_anchor_points(single_feature_map[0],feature_to_input_x_scale,feature_to_input_y_scale,feature_to_input_x_offset, feature_to_input_y_offset)

    print("finished")






 