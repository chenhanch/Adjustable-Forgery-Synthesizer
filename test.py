import os
import cv2
import json
import numpy as np
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from dwt_blending import blend_img_weight
import sys
sys.path.append('./library/')

from library.bi_online_generation import random_get_hull, random_erode_dilate, colorTransfer, blendImages


def load_landmarks(landmarks_file):
    """

    :param landmarks_file: input landmarks json file name
    :return: all_landmarks: having the shape of 64x2 list. represent left eye,
                            right eye, noise, left lip, right lip
    """
    all_landmarks = OrderedDict()
    with open(landmarks_file, "r", encoding="utf-8") as file:
        line = file.readline()
        while line:
            line = json.loads(line)
            all_landmarks[line["image_name"]] = np.array(line["landmarks"])
            line = file.readline()
    return all_landmarks


all_landmarks = load_landmarks("./images/000.mp4.json")
background_face = cv2.cvtColor(cv2.imread("./images/000_0000.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
foreground_face = cv2.cvtColor(cv2.imread("./images/231_0117.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
background_face = cv2.resize(background_face, (256, 256))
foreground_face = cv2.resize(foreground_face, (256, 256))

# get random type of initial blending mask    
mask = random_get_hull(all_landmarks['0000.png'], foreground_face)

#  random deform mask
distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.14, 0.15))])
mask = distortion.augment_image(mask)
mask = random_erode_dilate(mask)

# apply color transfer
foreground_face = colorTransfer(background_face, foreground_face, mask*255)

# constant blending 
blended_face1, mask = blendImages(foreground_face, background_face, mask*255)

# dynamic blending
_, blended_face2 = blend_img_weight(background_face.astype(np.float32), foreground_face.astype(np.float32), mask.astype(np.float32))
_, blended_face3 = blend_img_weight(background_face.astype(np.float32), foreground_face.astype(np.float32), mask.astype(np.float32))


plt.figure(figsize=(10,5)) 
plt.suptitle('example') 
plt.subplot(2,3,1), plt.title('I_t')
plt.imshow(background_face), plt.axis('off')
plt.subplot(2,3,2), plt.title('I_s')
plt.imshow(foreground_face), plt.axis('off')
plt.subplot(2,3,3), plt.title('Mask')
plt.imshow(mask, cmap='gray'), plt.axis('off')
plt.subplot(2,3,4), plt.title('I (alpha_i=1)')
plt.imshow(blended_face1.astype(np.uint8)), plt.axis('off')
plt.subplot(2,3,5), plt.title('I (random alpha_i)')
plt.imshow(blended_face2.astype(np.uint8)), plt.axis('off')
plt.subplot(2,3,6), plt.title('I (random alpha_i)')
plt.imshow(blended_face3.astype(np.uint8)), plt.axis('off')
plt.show()