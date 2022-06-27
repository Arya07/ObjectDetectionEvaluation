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
    'cleanser'       : 0,
    'banana'         : 1,
    'potted_meatcan' : 2,
    'scissors'       : 3,
    'crackerbox'     : 4,
    'powerdrill'     : 5,
    'sugarbox'       : 6,
    'mustard_bottle' : 7,
    'mug'            : 8,
    'pitcher'        : 9,
}


def add_prediction_to_array(image_id, prediction, array)
    ps = predictions.split('(')
    for i in range(1, len(ps), 2) # It reads all odds element of the array
        p = ps[i].split(' ')
        xmin  = float(p[0])
        ymin  = float(p[1])
        xmax  = float(p[2])
        ymax  = float(p[3])
        score = float(p[4])
        label = p[5]
        w = xmax - xmin
        h = ymax - ymin

        # Dictionary creation
        p_dict = {
            "image_id":    image_id
            "category_id": dict_cls_to_id[label],
            "bbox":        [xmin, ymin, w, h],
            "score":       score
        }
        array.appen(p_dict)

    return array


predictions_path =      ""
predictions_json_file = "predictions.json"
imageset_file =         ""

dirs_list = sorted(glob.glob(os.path.abspath(os.path.join(ho3d_path_train, '*'))))

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

for predictions_dir in dirs_list:
    print(predictions_dir)
    image_idx = -1
    for key, value in image_to_idx_dict:
        if predictions_dir in key:
            image_idx = value

    predictions_file = os.path.abspath(os.path.join(predictions_path, predictions_dir, 'data.log'))
    predictions = open(predictions_file, 'r')
    predictions_list = predictions.readlines()
    for prediction in predictions_list:
        predictions_array = add_prediction_to_array(image_idx, prediction, predictions_array)
        image_idx = image_idx + 1

with open(predictions_json_file, 'w') as json_file:
  json.dump(predictions_array, json_file)


