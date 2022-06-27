import cv2
import os
import glob
import numpy as np
from shutil import copyfile

annotations_path =      ""
annotations_json_file = "annotations.json"
imageset_file =         ""

ho3d_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Data', 'datasets', 'HO3D_V2'))
ho3d_path_Json = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Data', 'datasets', 'HO3D_V2_Json_format'))
if not os.path.exists(ho3d_path_Json):
    os.mkdir(ho3d_path_Json)

ho3d_path_train = os.path.abspath(os.path.join(ho3d_path, 'train'))
ho3d_path_train_iCWT = os.path.abspath(os.path.join(ho3d_path_Json, 'train'))
if not os.path.exists(ho3d_path_train_iCWT):
    os.mkdir(ho3d_path_train_iCWT)

target_images_dir = os.path.abspath(os.path.join(ho3d_path_train_iCWT, 'Images'))
if not os.path.exists(target_images_dir):
    os.mkdir(target_images_dir)
target_annotations_dir = os.path.abspath(os.path.join(ho3d_path_train_iCWT, 'Annotations'))
if not os.path.exists(target_annotations_dir):
    os.mkdir(target_annotations_dir)
target_masks_dir = os.path.abspath(os.path.join(ho3d_path_train_iCWT, 'Masks'))
if not os.path.exists(target_masks_dir):
    os.mkdir(target_masks_dir)
target_imagesets_dir = os.path.abspath(os.path.join(ho3d_path_train_iCWT, 'ImageSets'))
if not os.path.exists(target_imagesets_dir):
    os.mkdir(target_imagesets_dir)


dirs_list = sorted(glob.glob(os.path.abspath(os.path.join(ho3d_path_train, '*'))))

# Get list of images in the test set and associate each image to an id
imageset  = open(imageset_file, 'r')
imageset_lines = imageset.readlines()
image_to_idx_dict = OrderedDict()
idx = 0
for im_name in imageset_lines:
    image_to_idx_dict[im_name.strip()] = idx
    idx = idx + 1

dict_cls_to_id = {
    '021_bleach_cleanser' : 0,
    '011_banana'          : 1,
    '010_potted_meat_can' : 2,
    '037_scissors'        : 3,
    '003_cracker_box'     : 4,
    '035_power_drill'     : 5,
    '004_sugar_box'       : 6,
    '006_mustard_bottle'  : 7,
    '025_mug'             : 8
}

dir_names_to_class ={
    'ABF14':  '021_bleach_cleanser',
    'BB14':   '011_banana',
    'GPMF14': '010_potted_meat_can',
    'GSF14':  '037_scissors',
    'MC6':    '003_cracker_box',
    'MDF14':  '035_power_drill',
    'SiS1':   '004_sugar_box',
    'SM5':    '006_mustard_bottle',
    'SMu42':  '025_mug'
}

def create_categories_array(cls_to_id):

    categories_array = []
    
    for cls_name, cls_id in cls_to_id:
        category_dict = {
            "id"            : cls_id
            "name"          : cls_name
            "supercategory" : "ycb"
        }
        categories_array.append(category_dict)

    return categories_array

def create_images_array(image_to_idx):

    images_array = []

    for img_name, img_id in image_to_idx:
        image_dict = {
            "id"        : img_id
            "license"   : 1
            "file_name" : img_name + '.jpg' # TO CHECK
            "height"    : 480
            "width"     : 640
        }
        images_array.append(image_dict)

    return images_array

def create_annotations_array(array, img_name, objects_data, image_to_idx, cls_to_idx, annotations_id):

    img_id = image_to_idx[img_name]

    for obj in objects_data:
        if not obj['label'] == 'dummy':

            xmin = float(obj['xmin'])
            xmax = float(obj['xmax'])
            ymin = float(obj['ymin'])
            ymax = float(obj['ymax'])
            w = xmax - xmin
            h = ymax - ymin
            area = w*h

            object_dict = {
                "id"          : annotations_id,
                "image_id"    : img_id,
                "category_id" : cls_to_idx[obj['label']],
                "bbox"        : [xmin, ymin, w, h],
                "iscrowd"     : 0,
                "area"        : area,
            }
            annotations_id = annotations_id + 1
            array.append(object_dict)

    return array 

annotations_id    = 0
annotations_array = []
images_array      = create_images_array(image_to_idx_dict)
categories_array  = create_categories_array(dict_cls_to_id)

for dir in dirs_list:
    print(dir)
    found = False
    for k, v in dir_names_to_class.items():
        if k in dir:
            obj_class = v
            obj_dir = k
            found = True
            break
    if not found:
        print('Not found ' + dir)
        continue
    else:
        print('Found ' + dir)
        found = False

    segm_dir = os.path.abspath(os.path.join(dir, 'seg'))
    binary_segm_dir = os.path.abspath(os.path.join(target_masks_dir, obj_dir))
    if not os.path.exists(binary_segm_dir):
        os.mkdir(binary_segm_dir)

    annotations_dir = os.path.abspath(os.path.join(target_annotations_dir, obj_dir))
    if not os.path.exists(annotations_dir):
        os.mkdir(annotations_dir)

    images_dir = os.path.abspath(os.path.join(target_images_dir, obj_dir))
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)

    segm_files = sorted(glob.glob(os.path.abspath(os.path.join(segm_dir, '*'))))
    for segm_file in segm_files:
        img_name = segm_file.replace(segm_dir, '').replace('.png', '').replace('.jpg', '').replace('/', '')

        mask = cv2.resize(cv2.imread(segm_file), (640, 480))
        obj_indices = np.where((mask >= [100, 0, 0]).all(axis=2))

        # Compute segmentation binary masks
        binary_mask = np.zeros([mask.shape[0], mask.shape[1], 1], dtype=np.uint8)
        for i in range(len(obj_indices[0])):
            binary_mask[obj_indices[0][i], obj_indices[1][i]] = 255

        cv2.imwrite(os.path.abspath(os.path.join(binary_segm_dir, img_name + '.png')), binary_mask)

        # Compute annotations
        objects = []

        if len(obj_indices[0]) > 0:  #Check that the object is visible
            object = {}
            object['xmin'] = str(min(obj_indices[1]))
            object['ymin'] = str(min(obj_indices[0]))
            object['xmax'] = str(max(obj_indices[1])+1)
            object['ymax'] = str(max(obj_indices[0])+1)
            object['label'] = obj_class
            object['category'] = obj_class
            objects.append(object)
        annotations_array = create_annotations_array(annotations_array, img_name, objects, image_to_idx_dict, dict_cls_to_id, annotations_id)

        src_img_dir = os.path.abspath(os.path.join(dir, 'rgb', '%s.png'))
        # Copy images
        copyfile(src_img_dir%img_name, os.path.abspath(os.path.join(target_images_dir, obj_dir, img_name + '.png')))

annotations_dict = {
    "categories":  categories_array,
    "images":      images_array,
    "annotations": annotations_array
}

with open(annotations_json_file, 'w') as json_file:
  json.dump(annotations_dict, json_file)

