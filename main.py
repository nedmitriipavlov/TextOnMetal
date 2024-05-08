import cv2
import os
import time
import uuid
import numpy as np
import torch

from collections import namedtuple
from ultralytics import YOLO

from train import model

images_path = os.path.join('data', 'images')
cap = cv2.VideoCapture('video1.mp4')

ret, frame = cap.read()

while ret:
    imgname = os.path.join(images_path, 'square'+'.'+str(uuid.uuid1())+'.jpg')

    cv2.imwrite(imgname, frame)
    cv2.imshow('Image Collection', frame)

    ret, frame = cap.read()



# laplas_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
#
#
# def pixels_cals(image):
#     height, width = image.shape[:2]
#     white, black = 0, 0
#     for x in range(width):
#         for y in range(height):
#             pixel_val = image[y, x]
#             if np.all(np.equal(pixel_val, np.array([255, 255, 255], dtype=np.uint8))):
#                 black += 1
#             else:
#                 white += 1
#     return white, black
#
#
# def image_preproccessing(image):
#     # im = cv2.filter2D(image, -80, laplas_kernel)
#     # ret_, im_ = cv2.threshold(im, 1, 255, cv2.THRESH_TOZERO)
#     # bw_im = cv2.cvtColor(im_, cv2.COLOR_BGR2GRAY)
#     return image
#
#
# parent_dir = 'C:/Users/javas/PycharmProjects/TextOnMetal'
# model_path = 'C:/Users/javas/runs/detect/train2/weights/last.pt'
# directory = 'Train'
# n = 0
# path = None
# currentFrame = 0
# ret, frame = cap.read()
# flag_start = True
# Frame = namedtuple('Frame', 'img w b')
#
#
# out = cv2.VideoWriter('video_new.mp4', 1, -1, (frame.shape[1], frame.shape[0]))


    # if flag_start:
    #     img = image_preproccessing(frame)
    #     best_frame = Frame(img, *pixels_cals(img))
    #     flag_start = False
    #     path = os.path.join(parent_dir, f'{directory}{n}')
    #     while f'{directory}{n}' in os.listdir(parent_dir):
    #         n += 1
    #         path = os.path.join(parent_dir, f'{directory}{n}')
    #     os.mkdir(path)

    # print(1)
    # if ret:
    #     img = image_preproccessing(frame)
    #     # frame = Frame(frame, *pixels_cals(frame))
    #     # if frame.w >= best_frame.w and frame.b <= best_frame.b:
    #     #     cv2.imwrite(f'{path}/frame.jpg', img)
    #     cv2.imwrite(f'{path}/fr1ame{len(os.listdir(path))}.jpg', img)

