import os, random, json
import numpy as np
# import pandas as pd

import cv2
import matplotlib.pyplot as plt

def denormalize(point_array, img_shape):
    for point in point_array:
        point[0] = (point[0]/100)*img_shape[1]
        point[1] = (point[1]/100)*img_shape[0]
    point_array = np.array(point_array, dtype=np.int32)
    
    return point_array

def draw_mask_on_image(image, mask_json, opacity=1):
    width = image.shape[0]
    height = image.shape[1]
    mask = np.zeros(image.shape, dtype=np.uint8)
    
    for i, mask_object in enumerate(mask_json):
        if 'points' in mask_object['value']:
            points = mask_object['value']['points']
            denorm_points = denormalize(points, (width, height))
            mask = cv2.fillPoly(mask, [denorm_points], (random.randint(0,225),random.randint(0,225),random.randint(0,225)))
    
    masked_img = cv2.addWeighted(image, 1, mask, opacity, 0)
    return masked_img
    

def draw_bb_and_mask_on_image(image, mask_json, opacity=1):  
    width = image.shape[0]
    height = image.shape[1]
    mask = np.zeros(image.shape, dtype=np.uint8)
    
    for i, mask_object in enumerate(mask_json):
        if 'points' in mask_object['value']:
            points = mask_object['value']['points']
            denorm_points = denormalize(points, (width, height))
            
            #Create mask
            mask = cv2.fillPoly(mask, [denorm_points], (random.randint(0,200),random.randint(0,200),random.randint(0,200)))
            
            #Add bounding box to mask
            x,y,w,h = cv2.boundingRect(denorm_points)
            mask = cv2.rectangle(mask, (x,y), (x+w,y+h), (255,0,0), 2)
            
            #Add text label to mask
            label = mask_object['value']['polygonlabels'][0]          
            (w,h), _ =cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            mask = cv2.rectangle(mask, (x, y - 20), (x + w, y), (0,0,0), -1)
            mask = cv2.putText(mask, label, (x,y-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(225,225,225))
            
    
    masked_img = cv2.addWeighted(image, 1, mask, opacity, 0)
    return masked_img


def draw_mask_on_image_array(image_path, meta_data_json_path, opacity=1):
    n_images = len(os.listdir(image_path))
    figure, ax = plt.subplots(nrows=n_images, ncols=2, figsize=(100,100))   
    
    for ind, image in enumerate(os.listdir(image_path)):
        img = cv2.imread(image_path + image, -1)
        mask_json = json.load(open(meta_data_json_path + '{}.json'.format(ind+1)))
        
        masked_img = draw_mask_on_image(img, mask_json, opacity=opacity)
        
        mask_json = json.load(open(meta_data_json_path + '{}.json'.format(ind+1)))
        bbox_masked_img = draw_bb_and_mask_on_image(img, mask_json, opacity=opacity)
    
        ax[ind][0].imshow(masked_img)
        ax[ind][1].imshow(bbox_masked_img)
        
    plt.show()