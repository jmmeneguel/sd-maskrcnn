import numpy as np
import os, sys
from mrcnn.config import Config

class MaskConfig(Config):
  """Configuration for training on the toy shapes dataset.
  Derives from the base Config class and overrides values specific
  to the toy shapes dataset.
  """
  # Give the configuration a recognizable name
  NAME = "res50_update"

  # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
  # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
  # GPU_COUNT = 1
  # IMAGES_PER_GPU = 4

  # Number of classes (including background)
  NUM_CLASSES = 1 + 1  # background + object

  # Use small images for faster training. Set the limits of the small side
  # the large side, and that determines the image shape.
  IMAGE_MIN_DIM = 512
  IMAGE_MAX_DIM = 512

  # BACKBONE="resnet50"

  # Use smaller anchors because our image and objects are small
  # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

  # Reduce training ROIs per image because the images are small and have
  # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
  #TRAIN_ROIS_PER_IMAGE = 32

  # Use a small epoch since the data is simple
  # STEPS_PER_EPOCH = 8000/IMAGES_PER_GPU

  # use small validation steps since the epoch is small
  #VALIDATION_STEPS = 50

  # DETECTION_MIN_CONFIDENCE = 0.99

  def __init__(self, config):
    # Overriding things here.
    for x in config:
      setattr(self, x.upper(), config[x])
    super(MaskConfig, self).__init__()