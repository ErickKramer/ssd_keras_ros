import os
import numpy as np
from importlib import import_module

from rospkg import RosPack
from cv_bridge import CvBridge

from mas_perception_libs import ImageDetector, ImageDetectionKey
from mas_perception_libs.utils import process_image_message


class SSDKerasObjectDetector(ImageDetector):
    def __init__(self, **kwargs):
        # for ROS image message conversion
        self._cv_bridge = CvBridge()
        # initialize members
        self._target_size = None
        self._img_preprocess_func = None
        self._conf_threshold = None
        self._model = None
        self._rp = RosPack()
        super(SSDKerasObjectDetector, self).__init__(**kwargs)

    def load_model(self, **kwargs):
        from keras import backend as K
        from keras.optimizers import Adam
        from ssd_keras.keras_loss_function.keras_ssd_loss import SSDLoss

        # get path to weight file, relative to ROS package
        weight_file_package = kwargs.get('weight_file_package', None)
        if weight_file_package is None:
            raise ValueError('weight_file_package not specified.')
        weight_file_path = kwargs.get('weight_file_path', None)
        if weight_file_path is None:
            raise ValueError('weight_file_path not specified.')
        pkg_path = self._rp.get_path(weight_file_package)
        weight_file = os.path.join(pkg_path, weight_file_path)

        # get image size
        target_height = kwargs.get('target_height', None)
        if target_height is None:
            raise ValueError('target_height not specified.')
        target_width = kwargs.get('target_width', None)
        if target_width is None:
            raise ValueError('target_width not specified.')
        self._target_size = (target_height, target_width)

        # get confidence filter, will skip boxes with confidence less than this value
        self._conf_threshold = kwargs.get('conf_threshold', 0.5)

        # import get_model function
        get_model_func_name = kwargs.get('get_model_func_name', None)
        if get_model_func_name is None:
            raise ValueError('get_model_func_name not specified.')
        get_model_module = kwargs.get('get_model_module', None)
        if get_model_module is None:
            raise ValueError('get_model_module not specified.')
        func_get_model = getattr(import_module(get_model_module), get_model_func_name)

        K.clear_session()
        # call get_model function with image dimensions, number of classes & confidence threshold
        self._model = func_get_model(self._target_size + (3,), len(self._classes), self._conf_threshold)
        self._model.load_weights(weight_file, by_name=True)

        # Compile the model.
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self._model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
        # see https://github.com/keras-team/keras/issues/6462
        self._model._make_predict_function()

    def detect(self, image_messages):
        if len(image_messages) == 0:
            return []

        np_images = [process_image_message(msg, self._cv_bridge, self._target_size, self._img_preprocess_func)
                     for msg in image_messages]
        image_array =  np.stack(np_images, axis=0)

        y_pred = self._model.predict(image_array)
        if len(np_images) != y_pred.shape[0]:
            raise ValueError('number of predictions does not match number of images')

        # assuming y_pred entries to be [class, confidence, xmin, ymin, xmax, ymax]
        # filter predictions over a certain threshold of confidence
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > self._conf_threshold] for k in range(y_pred.shape[0])]

        predictions = []
        for i in range(len(np_images)):
            boxes = []
            for box in y_pred_thresh[i]:
                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                box_dict = {ImageDetectionKey.CLASS: self._classes[int(box[0])], ImageDetectionKey.CONF: box[1],
                            ImageDetectionKey.X_MIN: box[2] * image_messages[i].width / self._target_size[1],
                            ImageDetectionKey.Y_MIN: box[3] * image_messages[i].height / self._target_size[0],
                            ImageDetectionKey.X_MAX: box[4] * image_messages[i].width / self._target_size[1],
                            ImageDetectionKey.Y_MAX: box[5] * image_messages[i].height / self._target_size[0]}

                boxes.append(box_dict)

            predictions.append(boxes)

        return predictions
