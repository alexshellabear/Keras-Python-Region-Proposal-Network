from os import stat
import cv2
import numpy as np
try:
    from .utility_functions import pre_process_image_for_vgg
except:
    print("you should not be executing the program from this python file...")

class dataset_generator():
    def __init__(self,config):
        """
            Stores the settings functions required for a dataset
            Initialises
        """
        self.vggmodel = config["VGG16Model"]
        self.model_input_size = config["ModelInputSize"]
        self.dataset = []

    def save_dataset(self):
        """
            Saves the dataset as a pickle file
        """

    def convert_video_to_data(self,video_file_path):
        """
            Converts a video into a dataset which can be read by a human or the ML model

            Checks that the video file can be read
            resizes the image to suit the model
            gets a mask of the object - A1
            Gets the feature map by passing the input image through the CNN backbone, in this case vgg16
            Finds conversions between the input image space and feature map space
            checks whether or not an object was found
            goes through each anchor point, creates a mask for that box and finds the iou with the box and object mask
            gets highest iou and gets the coordinate of the anchor point
            Creates the ML output matrix
            Displays images for debugging
            Saves data in a list as an array in a human and machine readable format

            Assumption 1: There is only [0,1] objects in each frame
            Assumption 2: There is only one anchor box per anchor point

            TODO 1: Account for no objects, half done
        """
        cap = cv2.VideoCapture(video_file_path)
        assert cap.isOpened() # can open file
        
        index = -1
        while True:
            returned_value, frame = cap.read()
            if not returned_value:
                print("Can't receive frame (potentially stream end or end of file?). Exiting ...")
                break
            index += 1

            resized_frame = cv2.resize(frame,self.model_input_size)
            
            final_mask, final_result, object_identified = get_red_box(resized_frame)

            prediction_ready_image = pre_process_image_for_vgg(frame,self.model_input_size)
            feature_map = self.vggmodel.predict(prediction_ready_image)

            feature_to_input, input_to_feature = get_conversions_between_input_and_feature(prediction_ready_image.shape,feature_map.shape)
            coordinates_of_anchor_boxes = get_input_coordinates_of_anchor_points(feature_map.shape,feature_to_input)

            anchor_point_overlay_display_img = final_result.copy()
            if object_identified == True:
                iou_list = []
                
                for coord in coordinates_of_anchor_boxes:
                    anchor_box_mask = self.create_anchor_box_mask_on_input(coord,feature_to_input,final_mask.shape)
                    iou_list.append(self.get_iou_from_masks(final_mask, anchor_box_mask))
                    
                    self.draw_anchor_point_and_boxes(anchor_point_overlay_display_img,coord,feature_to_input)

                matching_anchor_box_index = iou_list.index(max(iou_list))
                matching_coord = coordinates_of_anchor_boxes[matching_anchor_box_index]
                cv2.circle(anchor_point_overlay_display_img,(matching_coord["x"],matching_coord["y"]),3,(255,255,255))

                output_shape = (feature_map.shape[1],feature_map.shape[2],1)
                ground_truth_output = np.zeros(output_shape,dtype=np.float64)

                coord_in_f_map = self.convert_input_image_coord_to_feature_map(matching_coord,input_to_feature)
                ground_truth_output[coord_in_f_map['y'],coord_in_f_map['x']] = [1.0]
            else:
                ground_truth_output = np.zeros(output_shape,dtype=np.float64)

            debug_image = self.gen_debug_image_and_display(resized_frame,final_mask,final_result,anchor_point_overlay_display_img,ground_truth_output)
            self.dataset.append({ 
                "Meta": {
                    "VideoPath" : video_file_path
                    ,"FrameIndex" : index
                }
                ,"MachineFormat" : {
                    "Input" : feature_map
                    ,"Ouput" : np.array([ground_truth_output])
                }
                ,"HumanFormat" : { 
                    "InputImage" : resized_frame
                    ,"ObjectMask" : final_mask
                    ,"MatchedCoord" : matching_coord
                    ,"ObjectDetected" : object_identified
                    ,"AllImagesSideBySide" : debug_image
                }
                })

            print(f"[{index}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}] max iou={max(iou_list)}, coord {iou_list.index(max(iou_list))}")

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def gen_debug_image_and_display(resized_frame,final_mask,final_result,anchor_point_overlay_display_img,ground_truth_output,wait_time_ms = 50):
        """
            For debugging purposes to display all the images and returns the concatenated result
            Convert everything to colour to be compatible to be concatenated
            Note 1: The final_mask is a 2d array that must be converted into a 3d array with 3 channels
            ground_truth_output_test = cv2.resize(ground_truth_output,final_mask.shape)
            cv2.imshow("ground_truth_output_test",ground_truth_output_test)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        """

        white_colour_image = np.ones(resized_frame.shape,dtype=np.uint8) * 255
        final_mask_colour = cv2.bitwise_and(white_colour_image,white_colour_image,mask=final_mask)

        debug_image = np.concatenate((resized_frame, final_mask_colour), axis=1)
        debug_image = np.concatenate((debug_image, final_result), axis=1)
        debug_image = np.concatenate((debug_image, anchor_point_overlay_display_img), axis=1)

        ground_truth_output_scaled = ground_truth_output * 255 
        ground_truth_output_cv2_uint8 = ground_truth_output_scaled.astype(np.uint8)
        ground_truth_output_cv2_mask = cv2.resize(ground_truth_output_cv2_uint8,final_mask.shape)
        ground_truth_output_colour = cv2.bitwise_and(white_colour_image,white_colour_image,mask=ground_truth_output_cv2_mask)
        
        debug_image = np.concatenate((debug_image, ground_truth_output_colour), axis=1)
        
        cv2.imshow("debug_image",debug_image)
        cv2.waitKey(wait_time_ms)
        return debug_image

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def draw_anchor_point_and_boxes(img_to_draw_on,anchor_point_coord,feature_to_input,scale=None,aspect_ratio=None):
        """
            For debugging to see the input image with the anchor points and boxes drawn 
            TODO: Account for scale and aspect ratio
        """
        cv2.circle(img_to_draw_on,(anchor_point_coord["x"],anchor_point_coord["y"]),2,(0,0,255))

        x1 = int(round(anchor_point_coord["x"] - feature_to_input["x_offset"]))
        y1 = int(round(anchor_point_coord["y"] - feature_to_input["y_offset"]))
        x2 = int(round(anchor_point_coord["x"] + feature_to_input["x_offset"]))
        y2 = int(round(anchor_point_coord["y"] + feature_to_input["y_offset"]))

        cv2.rectangle(img_to_draw_on,(x1,y1),(x2,y2),(255,255,255),1)

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def create_anchor_box_mask_on_input(coord,feature_to_input,mask_shape,scale=None,aspect_ratio=None):
        """
            Creates a mask where that covers the anchor box. This is used to then find the IOU of a blob
            TODO: Account for scale and aspect ratio
        """
        assert len(mask_shape) == 2 # Should be an image [width,height] because it is a mask

        # Create empty mask
        anchor_box_mask = np.zeros(mask_shape, dtype=np.uint8)
        
        x1 = int(round(coord["x"] - feature_to_input["x_offset"]))
        y1 = int(round(coord["y"] - feature_to_input["y_offset"]))
        x2 = int(round(coord["x"] + feature_to_input["x_offset"]))
        y2 = int(round(coord["y"] + feature_to_input["y_offset"]))

        fill_constant = -1
        anchor_box_mask = cv2.rectangle(anchor_box_mask,(x1,y1),(x2,y2),255,fill_constant)

        return anchor_box_mask

    #@staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def get_iou_from_masks(self,single_blob_mask_1, single_blob_mask_2): # TODO find variable
        """
            Gets the Intersection Over Area, aka how much they cross over divided by the total area
            from masks (greyscale images)

            Uses bitwise or to create new mask for union blob
            Uses bitwise and to create new mask for intersection blob

            Assumption 1: The masks must be greyscale
            Assumption 2: There must only be one blob (aka object) in each mask
            Assumption 3: Both masks must be the same dimensions (aka same sized object)
            Note 1: If the union area is 0, there are no blobs hence the IOU should be 0
        """
        assert len(single_blob_mask_1.shape) == 2 # Should be a greyscale image
        assert len(self.get_area_of_blobs(single_blob_mask_1)) == 1 # Mask should only have one blob in it
        assert len(single_blob_mask_2.shape) == 2 # Should be a greyscale image
        assert len(self.get_area_of_blobs(single_blob_mask_2)) == 1 # Mask should only have one blob in it
        assert single_blob_mask_1.shape[0] == single_blob_mask_2.shape[0] and single_blob_mask_1.shape[1] == single_blob_mask_2.shape[1]

        union_mask = cv2.bitwise_or(single_blob_mask_1,single_blob_mask_2)
        if len(self.get_area_of_blobs(union_mask)) == 1:
            union_area = self.get_area_of_blobs(union_mask)[0]
        else: 
            intersection_over_union = 0.0 # Stop math error, divide by 0
            return intersection_over_union

        intersection_mask = cv2.bitwise_and(single_blob_mask_1,single_blob_mask_2)
        if len(self.get_area_of_blobs(intersection_mask)) == 1:
            intersection_area = self.get_area_of_blobs(intersection_mask)[0]
        else: 
            intersection_area = 0.0

        intersection_over_union = intersection_area / union_area
        assert intersection_over_union >= 0.0
        assert intersection_over_union <= 1.0

        return intersection_over_union

    @staticmethod
    def get_area_of_blobs(mask):
        """
            Takes a cv2 mask, converts it to blobs and then finds the area and returns the blobs and the corresponding area
        """
        contours, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blob_areas = [cv2.contourArea(blob) for blob in contours]
        return blob_areas

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def convert_input_image_coord_to_feature_map(coord_in_input_space,input_to_feature):
        """
            Converts an point in the input image space to the feature map space returns as a dictionary

            Assumption 1: coord_in_input_space is a dictionary of {"x",int,"y",int}
            Assumption 2: input_to_features is a dictionary
        """
        x = int(round((coord_in_input_space["x"] + input_to_feature["x_offset"])*input_to_feature["x_scale"]))
        y = int(round((coord_in_input_space["y"] + input_to_feature["y_offset"])*input_to_feature["y_scale"]))
        coord_in_feature_map = {"x":x,"y":y}
        return coord_in_feature_map

def get_red_box(resized_frame,threshold_area = 400):
    """
        Uses HSV colour space to determine if a colour is actually red.
            Does this by considering the lower and upper colour space.
            Adds those two masks together
            Uses morphology to fill in the small gaps
            finds those blobs that have an area greater than threshold_area
            returns the overlayed image and the mask with only blobs greater than the threshold_area

            TODO refeactor code with new general utility functions, store away in other module for better use next timme
    """
    hsv_colour_img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    #lower red
    lower_red = np.array([0,110,110])
    upper_red = np.array([10,255,255])

    #upper red
    lower_red2 = np.array([170,50,50])
    upper_red2 = np.array([180,255,255])

    mask = cv2.inRange(hsv_colour_img, lower_red, upper_red)
    mask2 = cv2.inRange(hsv_colour_img, lower_red2, upper_red2)

    combined_mask = cv2.bitwise_or(mask,mask2)
    
    # Generate intermediate image; use morphological closing to keep parts of the brain together
    morphed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    final_mask = mask = np.zeros(mask.shape, dtype=np.uint8)

    contours, _  = cv2.findContours(morphed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs_greater_than_threshold = [blob for blob in contours if cv2.contourArea(blob) > threshold_area]
    for blob in blobs_greater_than_threshold:
        cv2.drawContours(final_mask, [blob], -1, (255), -1)

    final_result = cv2.bitwise_and(resized_frame,resized_frame, mask= final_mask)

    object_identified = final_mask.max() > 0

    return final_mask, final_result, object_identified


def get_conversions_between_input_and_feature(pre_processed_input_image_shape,feature_map_shape):
    """
        Finds the scale and offset from the feature map (output) of the CNN classifier to the pre-processed input image of the CNN
        Finds the inverse, pre-processed input image to feature map

        Input:
            pre_processed_input_image_shape: The 3/4d shape of the pre-processed input image that is passed to the backbone CNN classifier

        Returns a dictionary of values to easily pass variables
    """
    # Find shapes of feature maps and input images to the classifier CNN
    assert len(pre_processed_input_image_shape) in [3,4] # Either a 4d array with [:,height,width,channels] or just a single image [height,width,channels]
    assert len(feature_map_shape) in [3,4] # Either a 4d array with [:,height,width,channels] or just a single feature map [height,width,channels]
    if len(pre_processed_input_image_shape) == 3:
        img_height, img_width, _ = pre_processed_input_image_shape
    elif len(pre_processed_input_image_shape) == 4:
        _, img_height, img_width, _ = pre_processed_input_image_shape

    if len(feature_map_shape) == 3:
        features_height, features_width, _ = feature_map_shape
    elif len(feature_map_shape) == 4:
        _, features_height, features_width, _ = feature_map_shape

    # Find mapping from features map (output of vggmodel.predict) back to the input image
    feature_to_input_x_scale = img_width / features_width
    feature_to_input_y_scale = img_height / features_height

    # Put anchor points in the centre of 
    feature_to_input_x_offset = feature_to_input_x_scale/2
    feature_to_input_y_offset = feature_to_input_y_scale/2

    # Store as dictionary
    feature_to_input = {
        "x_scale": feature_to_input_x_scale
        ,"y_scale": feature_to_input_y_scale
        ,"x_offset" : feature_to_input_x_offset
        ,"y_offset" : feature_to_input_y_offset
    }

    # Find conversions from input image to feature map (CNN output)
    input_to_feature_x_scale = 1/feature_to_input_x_scale
    input_to_feature_y_scale = 1/feature_to_input_y_scale
    input_to_feature_x_offset = -feature_to_input_x_offset
    input_to_feature_y_offset = -feature_to_input_y_offset

    # Store as dictionary
    input_to_feature = {
        "x_scale": input_to_feature_x_scale
        ,"y_scale": input_to_feature_y_scale
        ,"x_offset" : input_to_feature_x_offset
        ,"y_offset" : input_to_feature_y_offset
    }

    return feature_to_input, input_to_feature

def get_input_coordinates_of_anchor_points(feature_map_shape,feature_to_input):
    """
        Maps the CNN output (Feature map) coordinates on the pre-processed input image space to the backbone CNN 
        Returns the coordinates as a flattened list of dictionaries with the format {"x":x,"y":y}
    """
    assert len(feature_map_shape) in [3,4] # Either a 4d array with [:,height,width,channels] or just a single feature map [height,width,channels]

    if len(feature_map_shape) == 3:
        features_height, features_width, _ = feature_map_shape
    elif len(feature_map_shape) == 4:
        _, features_height, features_width, _ = feature_map_shape

    # For the feature map (x,y) determine the anchors on the input image (x,y) as array 
    feature_to_input_coords_x  = [int(x_feature*feature_to_input["x_scale"]+feature_to_input["x_offset"]) for x_feature in range(features_width)]
    feature_to_input_coords_y  = [int(y_feature*feature_to_input["y_scale"]+feature_to_input["y_offset"]) for y_feature in range(features_height)]
    coordinate_of_anchor_points = [{"x":x,"y":y} for x in feature_to_input_coords_x for y in feature_to_input_coords_y]

    return coordinate_of_anchor_points
