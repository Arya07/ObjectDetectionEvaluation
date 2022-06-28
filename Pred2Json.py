import os
import json
import glob
from collections import OrderedDict

# Predictions
# Folder:
#    - Folder_pred0: 
#        - Pred0.txt
#    - Folder_pred1: 
#        - Pred1.txt
#    .....

dict_cls_to_id = {
    'bleach'         : 0,
    'banana'         : 1,
    'pottedmeat'     : 2,
    'scissors'       : 3,
    'crackerbox'     : 4,
    'driller'        : 5,
    'sugarbox'       : 6,
    'mustard'        : 7,
    'mug'            : 8,
    'pitcher'        : 9
}


def add_prediction_to_array(image_id, prediction, array):
    if len(prediction.split('((')) == 2:
        return array
    ps = prediction.split('(')
    for i in range(1, len(ps), 2): # It reads all odds element of the array
        p = ps[i].split(' ')
        xmin  = float(p[0])
        ymin  = float(p[1])
        xmax  = float(p[2])
        ymax  = float(p[3])
        score = float(p[4])
        label = p[6]
        w = xmax - xmin
        h = ymax - ymin

        # Dictionary creation
        p_dict = {
            "image_id":    image_id,
            "category_id": dict_cls_to_id[label],
            "bbox":        [xmin, ymin, w, h],
            "score":       score
        }
        array.append(p_dict)

    return array


predictions_path =      "/home/icub/Users/emaiettini/ObjectDetectionEvaluation/yarp_predictions/clean_dispblobber_00001"
predictions_json_file = "predictions_clean_dispblobber_00001.json"
imageset_file =         "/home/icub/Users/emaiettini/ObjectDetectionEvaluation/HO3D_V2_Json_format/train/ImageSets/imageset_test.txt"



# Get list of images in the test set and associate each image to an id
imageset  = open(imageset_file, 'r')
imageset_lines = imageset.readlines()
image_to_idx_dict = OrderedDict()
idx = 0
for im_name in imageset_lines:
    image_to_idx_dict[im_name.strip()] = idx
    idx = idx + 1

# Initialize the array of predictions
predictions_array = []

predictions_file = os.path.abspath(os.path.join(predictions_path, 'data.log'))
predictions = open(predictions_file, 'r')
predictions_list = predictions.readlines()
image_idx = 0
for prediction in predictions_list:
    predictions_array = add_prediction_to_array(image_idx, prediction, predictions_array)
    image_idx = image_idx + 1

with open(predictions_json_file, 'w') as json_file:
  json.dump(predictions_array, json_file)


