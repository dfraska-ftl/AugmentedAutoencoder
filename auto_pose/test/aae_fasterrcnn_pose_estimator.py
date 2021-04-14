import cv2
import numpy as np
import glob
import os
import configparser
from collections import namedtuple
from auto_pose.ae import factory, utils
import tensorflow.compat.v1 as tf

# import keras
# from keras_retinanet.models import load_model, backbone
# from keras_retinanet.models.retinanet import __build_anchors as build_anchors
# from keras_retinanet.models.retinanet import AnchorParameters
# from keras_retinanet import layers
# from keras_retinanet.utils.image import preprocess_image, resize_image
# from keras.backend.tensorflow_backend import set_session

class PoseEstimatorArgs:
    def __init__(self, test_configpath):
        test_args = configparser.ConfigParser()
        test_args.read(test_configpath)
        
        self.camPose = test_args.getboolean('CAMERA', 'camPose')
        self.camK = np.array(eval(test_args.get('CAMERA', 'K_test'))).reshape(3,3)
        self.width = test_args.getint('CAMERA', 'width')
        self.height = test_args.getint('CAMERA', 'height')
        
        self.upright = test_args.getboolean('AAE', 'upright')
        self.experiments = eval(test_args.get('AAE', 'experiments'))

        self.class_names = eval(test_args.get('DETECTOR', 'class_names'))
        self.det_threshold = eval(test_args.get('DETECTOR', 'det_threshold'))
        self.detector_model_path = str(test_args.get('DETECTOR', 'detector_model_path'))
        self.model_backbone = test_args.get('DETECTOR', 'backbone')
        self.nms_threshold = test_args.getfloat('DETECTOR','nms_threshold'),
        self.score_threshold = test_args.getfloat('DETECTOR','det_threshold'),
        self.max_detections = test_args.getint('DETECTOR', 'max_detections')
        
        self.per_process_gpu_memory_fraction = test_args.getfloat(
            'MODEL','gpu_memory_fraction')

class AeFasterRCNNPoseEstimator:
    def __init__(self, args: PoseEstimatorArgs, workspace_path):
        self._camPose = args.camPose
        self._camK = args.camPose
        self._width = args.width
        self._height = args.height
        
        self._upright = args.upright
        self.all_experiments = args.experiments

        self.class_names = args.class_names
        self.det_threshold = args.det_threshold

        self.all_codebooks = []
        self.all_train_args = []
        self.pad_factors = []
        self.patch_sizes = []

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction

        self.sess = tf.Session(config=config)
        set_session(self.sess)
        self.detector = load_model(
            args.detector_model_path, backbone_name=args.model_backbone)
        #detector = self._load_model_with_nms(test_args)

        for i, experiment in enumerate(self.all_experiments):
            full_name = experiment.split('/')
            experiment_name = full_name.pop()
            experiment_group = full_name.pop() if len(full_name) > 0 else ''
            log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
            ckpt_dir = utils.get_checkpoint_dir(log_dir)
            train_cfg_file_path = utils.get_train_config_exp_file_path(log_dir, experiment_name)
            print(train_cfg_file_path)
            # train_cfg_file_path = utils.get_config_file_path(workspace_path, experiment_name, experiment_group)
            train_args = configparser.ConfigParser()
            train_args.read(train_cfg_file_path)
            self.all_train_args.append(train_args)
            self.pad_factors.append(
                train_args.getfloat('Dataset','PAD_FACTOR'))
            self.patch_sizes.append((
                train_args.getint('Dataset','W'), train_args.getint('Dataset', 'H')))

            self.all_codebooks.append(factory.build_codebook_from_name(
                experiment_name, experiment_group, return_dataset=False))
            saver = tf.train.Saver(
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=experiment_name))
            factory.restore_checkpoint(self.sess, saver, ckpt_dir)

    def extract_square_patch(
            self, scene_img, bb_xywh, pad_factor,resize=(128,128),
            interpolation=cv2.INTER_NEAREST,black_borders=False):
        x, y, w, h = np.array(bb_xywh).astype(np.int32)
        size = int(np.maximum(h, w) * pad_factor)

        
        left = np.maximum(x+w//2-size//2, 0)
        right = x+w//2+size/2
        top = np.maximum(y+h//2-size//2, 0)
        bottom = y+h//2+size//2

        scene_crop = scene_img[top:bottom, left:right].copy()

        if black_borders:
            scene_crop[:(y-top),:] = 0
            scene_crop[(y+h-top):,:] = 0
            scene_crop[:,:(x-left)] = 0
            scene_crop[:,(x+w-left):] = 0

        scene_crop = cv2.resize(scene_crop, resize, interpolation = interpolation)

        return scene_crop

    def process_detection(self, color_img):
        ''' Performs 2d bounding box detection
        '''
        
        H, W = color_img.shape[:2]

        pre_image = preprocess_image(color_img)
        res_image, scale = resize_image(pre_image)

        batch_image = np.expand_dims(res_image, axis=0)
        print(batch_image.shape)
        print(batch_image.dtype)
        boxes, scores, labels = self.detector.predict_on_batch(batch_image)


        valid_dets = np.where(scores[0] >= self.det_threshold)

        boxes /= scale

        scores = scores[0][valid_dets]
        boxes = boxes[0][valid_dets]
        labels = labels[0][valid_dets]

        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for box,score,label in zip(boxes, scores, labels):
            box[0] = np.minimum(np.maximum(box[0],0),W)
            box[1] = np.minimum(np.maximum(box[1],0),H)
            box[2] = np.minimum(np.maximum(box[2],0),W)
            box[3] = np.minimum(np.maximum(box[3],0),H)

            bb_xywh = np.array([box[0],box[1],box[2]-box[0],box[3]-box[1]])
            if bb_xywh[2] < 0 or bb_xywh[3] < 0:
                continue

            filtered_boxes.append(bb_xywh)
            filtered_scores.append(score)
            filtered_labels.append(label)
        return (filtered_boxes, filtered_scores, filtered_labels)

    def process_pose(self, filtered_boxes, filtered_labels, color_img, camPose=None):
        ''' Finds the poses given the set of cropped images 
        '''
        all_pose_estimates = []
        all_class_idcs = []

        for box_xywh, label in zip(filtered_boxes, filtered_labels):
            H_est = np.eye(4)

            try:
                clas_idx = self.class_names.index(label)
            except:
                print('%s not contained in config class_names %s', (label, self.class_names))
                continue

            det_img = self.extract_square_patch(
                color_img, box_xywh, self.pad_factors[clas_idx],
                resize=self.patch_sizes[clas_idx],
                interpolation=cv2.INTER_LINEAR, black_borders=True)

            Rs_est, ts_est = self.all_codebooks[clas_idx].auto_pose6d(
                self.sess, det_img, box_xywh, self._camK, 1,
                self.all_train_args[clas_idx], upright=self._upright)

            R_est = Rs_est.squeeze()
            t_est = ts_est.squeeze()

            H_est[:3,3] = t_est
            H_est[:3,:3] = R_est
            print(f'translation from camera: {H_est[:3,3]}')

            if self._camPose:
                H_est = np.dot(camPose, H_est)           

            all_pose_estimates.append(H_est)
            all_class_idcs.append(clas_idx)

        return (all_pose_estimates, all_class_idcs)