# -*- coding: utf-8 -*-
import numpy as np
import os
import time
from tqdm import tqdm

from sd_maskrcnn import utils
from sd_maskrcnn.config import MaskConfig

from mrcnn import model as modellib, utils as utilslib

class SDMaskRCNNModel(object):

    def __init__(self, path, mode, config):
        
        if not os.path.exists(path):
            raise ValueError('No model located at {}'.format(path))
        if mode not in ['training', 'inference']:
            raise ValueError('Can only create a model with mode inference or training')
        
        self.path = path
        self.mode = mode
        self.config = config

        image_shape = self.config['settings']['image_shape']
        self.config['settings']['image_min_dim'] = min(image_shape)
        self.config['settings']['image_max_dim'] = max(image_shape)
        self._mconfig = MaskConfig(self.config['settings'])
        
        self._model = modellib.MaskRCNN(mode=self.mode, 
                                        config=self._mconfig,
                                        model_dir=self.path)

        if self.mode == 'training':
            exclude_layers = []
            weights_path = self.config['weights']
        
            # Select weights file to load
            if self.config['weights'].lower() == "coco":
                weights_path = os.path.join(self.config['path'], 'mask_rcnn_coco.h5')
                # Download weights file
                if not os.path.exists(weights_path):
                    utilslib.download_trained_weights(weights_path)
                if self.config['settings']['image_channel_count'] == 1:
                    exclude_layers = ['conv1']
                # Exclude the last layers because they require a matching
                # number of classes
                exclude_layers += ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
            elif self.config['weights'].lower() == "last":
                # Find last trained weights
                weights_path = self._model.find_last()
            elif self.config['weights'].lower() == "imagenet":
                # Start from ImageNet trained weights
                weights_path = self._model.get_imagenet_weights()
                if self.config['settings']['image_channel_count'] == 1:
                    exclude_layers = ['conv1']
        else:
            weights_path = self.path

        # Load weights
        if weights_path != 'new':
            print("Loading weights ", weights_path)
            model.load_weights(weights_path, by_name=True, exclude=exclude_layers)
        
    def detect(self, image, bin_mask=None, overlap_thresh=0.5):

        if self.mode != 'inference':
            print('Can only call detect in inference mode!')
            return None

        # Run detection
        r = self._model.detect([image], verbose=0)[0]

        # If we choose to mask out bin pixels, load the bin masks and
        # transform them properly.
        # Then, delete the mask, score, class id, and bbox corresponding
        # to each mask that is entirely bin pixels.
        if bin_mask is not None:

            deleted_masks = [] # which segmasks are gonna be tossed?
            num_detects = r['masks'].shape[2]
            for k in range(num_detects):
                # compute the area of the overlap.
                inter = np.logical_and(bin_mask, r['masks'][:,:,k])
                frac_overlap =  np.sum(inter) / np.sum(r['masks'][:,:,k])
                if frac_overlap <= overlap_thresh:
                    deleted_masks.append(k)

            r['masks'] = [r['masks'][:,:,k] for k in range(num_detects) if k not in deleted_masks]
            r['masks'] = np.stack(r['masks'], axis=2) if r['masks'] else np.array([])
            r['rois'] = [r['rois'][k,:] for k in range(num_detects) if k not in deleted_masks]
            r['rois'] = np.stack(r['rois'], axis=0) if r['rois'] else np.array([])
            r['class_ids'] = np.array([r['class_ids'][k] for k in range(num_detects)
                                       if k not in deleted_masks])
            r['scores'] = np.array([r['scores'][k] for k in range(num_detects)
                                       if k not in deleted_masks])
        
        import pdb; pdb.set_trace()
        masks = np.stack([r['masks'][:,:,i] for i in range(r['masks'].shape[2])]) if np.any(r['masks']) else np.array([])


        return r

    
    def detect_dataset(self, output_dir, dataset, bin_mask_dir=None, overlap_thresh=0.5):
        
        # Create subdirectory for prediction masks
        pred_dir = os.path.join(output_dir, 'pred_masks')
        utils.mkdir_if_missing(pred_dir)

        # Create subdirectory for prediction scores & bboxes
        pred_info_dir = os.path.join(output_dir, 'pred_info')
        utils.mkdir_if_missing(pred_info_dir)

        # Create subdirectory for transformed GT segmasks
        resized_segmask_dir = os.path.join(output_dir, 'modal_segmasks_processed')
        utils.mkdir_if_missing(resized_segmask_dir)

        # Feed images into model one by one. For each image, predict and save.
        image_ids = dataset.image_ids
        indices = dataset.indices

        print('MAKING PREDICTIONS')
        for image_id in tqdm(image_ids):
            # Load image and ground truth data and resize for net
            image, _, _, _, gt_mask =\
                modellib.load_image_gt(dataset, 
                                    self._mconfig, 
                                    image_id,
                                    use_mini_mask=False)

            bin_mask = None
            if bin_mask_dir is not None:
                name = 'image_{:06d}.png'.format(indices[image_id])
                bin_mask = io.imread(os.path.join(bin_mask_dir, name))[:,:,np.newaxis]
                bin_mask, _, _, _, _ = utilslib.resize_image(
                    bin_mask,
                    max_dim=inference_config.IMAGE_MAX_DIM,
                    min_dim=inference_config.IMAGE_MIN_DIM,
                    mode=inference_config.IMAGE_RESIZE_MODE
                )

                bin_mask = bin_mask.squeeze()
            
            r = self.detect(image, bin_mask, overlap_thresh)

            # Save copy of transformed GT segmasks to disk in preparation for annotations
            mask_name = 'image_{:06d}'.format(image_id)
            mask_path = os.path.join(resized_segmask_dir, mask_name)

            # save the transpose so it's (n, h, w) instead of (h, w, n)
            np.save(mask_path, gt_mask.transpose(2, 0, 1))

            # Save masks
            save_masks = np.stack([r['masks'][:,:,i] for i in range(r['masks'].shape[2])]) if np.any(r['masks']) else np.array([])
            save_masks_path = os.path.join(pred_dir, 'image_{:06d}.npy'.format(image_id))
            np.save(save_masks_path, save_masks)

            # Save info
            r_info = {
                'rois': r['rois'],
                'scores': r['scores'],
                'class_ids': r['class_ids']
            }
            r_info_path = os.path.join(pred_info_dir, 'image_{:06d}.npy'.format(image_id))
            np.save(r_info_path, r_info)


    def train(self, train_dataset, val_dataset):

        if self.mode != 'training':
            print('Can only call train in training mode!')
            return
        
        self._model.train(train_dataset, val_dataset, 
                          learning_rate=self._mconfig.LEARNING_RATE,
                          epochs=self.config['epochs'], layers='all')

        # save in the models folder
        current_datetime = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(self.config['path'], "sd_mask_rcnn_{}_{}.h5".format(self._mconfig.NAME, current_datetime))
        self._model.keras_model.save_weights(model_path)
