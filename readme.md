# Project Name: #
Keras Python Region Proposal Network

# Description: #
This project is to build a region proposal network (RPN) from scratch to develop bounding box object detection algorithm.

There are 4 key components
1) Generate ground truths: Label the images
2) Generate a dataset: Take ground truth labels and form a dataset from it
3) Transfer learning: Use pre-existing weights of VGG16 imagenet and customise it to this dataset
4) Predict: Use the trained model to make predictions on incoming images

# Environment set up: #
How to set up the right environment to run the program

1) Make sure you have python installed, use the following link https://www.python.org/downloads/
2) How to get started with python https://www.python.org/about/gettingstarted/
3) How to set up your requirements with requirements.txt https://stackoverflow.com/questions/7225900/how-can-i-install-packages-using-pip-according-to-the-requirements-txt-file-from
    
Okay you're now set with the right environment let's get this show on the road!

# Creating your own object detector: #
Make your current working directory when running scripts the same as the one readme.md (this file) is stored in
Psst you can check using os.getcwd()

1) Take a bunch of photos of your face and drop them into the path "1. Data Gen\1. Data"
2) Run the python script "1. Data Gen\create_ground_truth_bounding_box.py" this is a labelling tool that can enable you to label data.
- Assumption 1: Every image name is unique
- Assumption 2: There is only one object in a frame at a time
- Note 1: It will resize your images to fix the screen hence it will have much lower resolution so that I have to do less programming. 
- Note 2: It will run through all of those files that do not have a corresponding .csv file (hence they are assumed not to have been labelled). If you want to re-label them
3) Look at the tutorial image below to label images.

![Tutorial Image](https://github.com/alexshellabear/Face-Detection-RCNN/blob/master/1.%20Data%20Gen/Ground%20Truth%20Tutorial.png)

4) Now that all your images are labeled lets we'll need to generate a dataset of smaller sections of the image to train the image classifier on. Run the python script "1. Data Gen\\generate_data_set_labels.py" to select thousands of bounding boxes of interest based upon cv2's fast selective search function. This will have very few foreground labeled images (what i've specifed as an IOU of 90%). This script will take a while to run because it's not optimised.