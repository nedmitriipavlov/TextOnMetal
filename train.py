import cv2

import os
import numpy as np
import torch

from collections import namedtuple
from ultralytics import YOLO

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

