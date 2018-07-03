import numpy as np

from cv_bridge import CvBridge

from mdr_perception_libs import RgbDetector, RgbDetectionKey
from mdr_perception_libs.utils import process_image_message

from ssd_keras.keras_loss_function.keras_ssd_loss import SSDLoss


class SSDKerasObjectDetector(RgbDetector):
    def __init__(self, **kwargs):
        # for ROS image message conversion
        self._cv_bridge = CvBridge()
        # initialize members
        self._target_size = None
        self._img_preprocess_func = None
        self._conf_threshold = None
        self._classes = None
        self._model = None
        super(SSDKerasObjectDetector, self).__init__(**kwargs)

    @property
    def classes(self):
        return self._classes

    def load_model(self, **kwargs):
        from keras import backend as K
        from keras.optimizers import Adam

        weights_path = kwargs.get('weights_path', None)
        if weights_path is None:
            raise ValueError('weights_path not specified.')

        self._target_size = kwargs.get('target_size', None)
        if self._target_size is None:
            raise ValueError('target_size not specified.')

        self._conf_threshold = kwargs.get('conf_threshold', 0.5)

        func_get_model = kwargs.get('func_get_model', None)
        if func_get_model is None:
            raise ValueError('func_get_model not specified.')

        self._classes = kwargs.get('classes', None)
        if self._classes is None:
            raise ValueError('list of classes not specified.')

        K.clear_session()
        # call get_model function with image dimensions, number of classes & confidence threshold
        self._model = func_get_model(self._target_size + (3,), len(self._classes), self._conf_threshold)
        self._model.load_weights(weights_path, by_name=True)

        # Compile the model.
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self._model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

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
                box_dict = { RgbDetectionKey.CLASS: self._classes[int(box[0])], RgbDetectionKey.CONF: box[1] }

                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                box_dict[RgbDetectionKey.X_MIN] = box[2] * image_messages[i].width / self._target_size[1]
                box_dict[RgbDetectionKey.Y_MIN] = box[3] * image_messages[i].height / self._target_size[0]
                box_dict[RgbDetectionKey.X_MAX] = box[4] * image_messages[i].width / self._target_size[1]
                box_dict[RgbDetectionKey.Y_MAX] = box[5] * image_messages[i].height / self._target_size[0]

                boxes.append(box_dict)

            predictions.append(boxes)

        return predictions
