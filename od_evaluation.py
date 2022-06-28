# import the necessary packages
#from pyimagesearch.utils_map import run_inference
#from pyimagesearch.utils_map import load_yolo_cls_idx
#from pyimagesearch import config
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
#from darknet import darknet

# This functions expects to have a json file for the annotation and a json file for the predictions
#def compute_map():
COCO_GT_ANNOTATION = "annotations.json"
COCO_VAL_PRED = "predictions_clean_dispblobber_00001.json"
# use the COCO class to load and read the ground-truth annotations
cocoAnnotation = COCO(annotation_file=COCO_GT_ANNOTATION)

# load detection JSON file from the disk
cocovalPrediction = cocoAnnotation.loadRes(COCO_VAL_PRED)
# initialize the COCOeval object by passing the coco object with
# ground truth annotations, coco object with detection results
cocoEval = COCOeval(cocoAnnotation, cocovalPrediction, "bbox")	
# run evaluation for each image, accumulates per image results
# display the summary metrics of the evaluation
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()


