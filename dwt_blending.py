import numpy as np
from dwt_common import dwt, iwt, normalization
import random
import cv2

def blend_img_random_choice(background_face, foreground_face, final_mask):
    final_mask = cv2.resize(final_mask, (int(final_mask.shape[1]/2), int(final_mask.shape[0]/2)))
    # [x_LL, x_HL, x_LH, x_HH]   
    background_face_dwt = dwt(background_face)   
    foreground_face_dwt = dwt(foreground_face)   
    blending_image_dwt = np.zeros(background_face_dwt.shape, dtype=np.float32)
    flag = True
    while(flag):
        # x_LL
        if random.random() < 0.5:
            blending_image_dwt[:,:,0:3] = foreground_face_dwt[:,:,0:3] * final_mask + background_face_dwt[:,:,0:3] * (1.0 - final_mask)
            flag = False
        else:
            blending_image_dwt[:,:,0:3] = background_face_dwt[:,:,0:3]
        # x_HL
        if random.random() < 0.5:
            blending_image_dwt[:,:,3:6] = foreground_face_dwt[:,:,3:6] * final_mask + background_face_dwt[:,:,3:6] * (1.0 - final_mask)
            flag = False
        else:
            blending_image_dwt[:,:,3:6] = background_face_dwt[:,:,3:6]
        # x_LH
        if random.random() < 0.5:
            blending_image_dwt[:,:,6:9] = background_face_dwt[:,:,6:9]
            flag = False
        else:
            blending_image_dwt[:,:,6:9] = foreground_face_dwt[:,:,6:9] * final_mask + background_face_dwt[:,:,6:9] * (1.0 - final_mask)
        # x_HH
        if random.random() < 0.5:
            blending_image_dwt[:,:,9:] = background_face_dwt[:,:,9:]
            flag = False
        else:
            blending_image_dwt[:,:,9:] = foreground_face_dwt[:,:,9:] * final_mask + background_face_dwt[:,:,9:] * (1.0 - final_mask)
    blending_image = iwt(blending_image_dwt)
    return blending_image_dwt, blending_image


def blend_img_weight(background_face, foreground_face, final_mask):
    final_mask = cv2.resize(final_mask, (int(final_mask.shape[1]/2), int(final_mask.shape[0]/2)))
    # [x_LL, x_HL, x_LH, x_HH]   
    background_face_dwt = dwt(background_face)   
    foreground_face_dwt = dwt(foreground_face)   
    blending_image_dwt = np.zeros(background_face_dwt.shape, dtype=np.float32)
    # x_LL
    weight = np.random.uniform(0.5, 1.0, 1)
    blending_image_dwt[:,:,0:3] = foreground_face_dwt[:,:,0:3] * (weight * final_mask) + background_face_dwt[:,:,0:3] * (1.0 - (weight * final_mask))
    # x_HL
    weight = np.random.uniform(0.5, 1.0, 1)
    blending_image_dwt[:,:,3:6] = foreground_face_dwt[:,:,3:6] * (weight * final_mask) + background_face_dwt[:,:,3:6] * (1.0 - (weight * final_mask))
    # x_LH
    weight = np.random.uniform(0.5, 1.0, 1)
    blending_image_dwt[:,:,6:9] = foreground_face_dwt[:,:,6:9] * (weight * final_mask) + background_face_dwt[:,:,6:9] * (1.0 - (weight * final_mask))
    # x_HH
    weight = np.random.uniform(0.5, 1.0, 1)
    blending_image_dwt[:,:,9:] = foreground_face_dwt[:,:,9:] * (weight * final_mask) + background_face_dwt[:,:,9:] * (1.0 - (weight * final_mask))
    blending_image = iwt(blending_image_dwt)
    return blending_image_dwt, blending_image


def blend_img_weight_matrix(background_face, foreground_face, final_mask):
    final_mask = cv2.resize(final_mask, (int(final_mask.shape[1]/2), int(final_mask.shape[0]/2)))
    # [x_LL, x_HL, x_LH, x_HH]   
    background_face_dwt = dwt(background_face)   
    foreground_face_dwt = dwt(foreground_face)   
    blending_image_dwt = np.zeros(background_face_dwt.shape, dtype=np.float32)
    # x_LL
    weight = np.random.uniform(0.5, 1.0, final_mask.shape)
    blending_image_dwt[:,:,0:3] = foreground_face_dwt[:,:,0:3] * (weight * final_mask) + background_face_dwt[:,:,0:3] * (1.0 - (weight * final_mask))
    # x_HL
    weight = np.random.uniform(0.5, 1.0, final_mask.shape)
    blending_image_dwt[:,:,3:6] = foreground_face_dwt[:,:,3:6] * (weight * final_mask) + background_face_dwt[:,:,3:6] * (1.0 - (weight * final_mask))
    # x_LH
    weight = np.random.uniform(0.5, 1.0, final_mask.shape)
    blending_image_dwt[:,:,6:9] = foreground_face_dwt[:,:,6:9] * (weight * final_mask) + background_face_dwt[:,:,6:9] * (1.0 - (weight * final_mask))
    # x_HH
    weight = np.random.uniform(0.5, 1.0, final_mask.shape)
    blending_image_dwt[:,:,9:] = foreground_face_dwt[:,:,9:] * (weight * final_mask) + background_face_dwt[:,:,9:] * (1.0 - (weight * final_mask))
    blending_image = iwt(blending_image_dwt)
    return blending_image_dwt, blending_image
        
