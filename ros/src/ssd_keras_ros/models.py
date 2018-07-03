from ssd_keras.models.keras_ssd300 import ssd_300
from ssd_keras.eval_utils.coco_utils import get_coco_category_maps


def get_coco_classes(annotation_file):
    """
    parse COCO annotation file into dictionary mapping predicted number to class name

    :param annotation_file: JSON file as provided on http://cocodataset.org (i.e. instances_val2017.json)
    :return: dictionary { key = predicted int value, value = class name }
    """
    _, _, _, classes_to_names = get_coco_category_maps(annotation_file)
    # create dictionary of predicted_class_number : class_name, skipping background class
    classes = {k + 1: v for k, v in enumerate(classes_to_names[1:])}
    return classes


def get_ssd300_coco_inference_model(image_size, class_num, conf_threshold):
    return ssd_300(image_size=image_size,
                   n_classes=class_num,
                   mode='inference',
                   l2_regularization=0.0005,
                   # The scales for PASCAL VOC are [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
                   scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                   aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                            [1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5]],
                   two_boxes_for_ar1=True,
                   steps=[8, 16, 32, 64, 100, 300],
                   offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                   clip_boxes=False,
                   variances=[0.1, 0.1, 0.2, 0.2],
                   normalize_coords=True,
                   subtract_mean=[123, 117, 104],
                   swap_channels=[2, 1, 0],
                   confidence_thresh=conf_threshold,
                   iou_threshold=0.45,
                   top_k=200,
                   nms_max_output_size=400)
