# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:28:19 2022

@author: Elijah Gallagher
"""
import cv2
import cv
import os
import numpy as np
import matplotlib.pyplot as plt

def reshapeImg(path):
    
    path = readImg(path)
    
    
    #Load in image from path gotten by readImg
    img = cv2.imread(path, 1)
    #Some scaling to make it fit a model
    
    #img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_NEAREST)
    #confirmed this alters the image to a 32,32,3 format which is prefered for a colored image.
  #  plt.imshow(img, cmap=plt.cm.binary)
  #  plt.show()
    
    #Grab color info (important because our model cannot take a (32,32,1) array)
    COLORED = False
    if img.shape[2] > 2:
        COLORED = True
    #return reshaped img and bool for greyscale/RGB
    return(img, COLORED)



def reshapeAllImg(path='photos'):
    #this function takes a path (folder you want to pull from) and returns all the images in an array
    #they will all be downscaled to 32x32x3
    #NOTE: this currently does not save them to any location because i was constantly changing the images i was using so-
    #you should be aware every time you call this it will need to load ALL the images and downscale them.
    
    reshaped_imgs = []
    imgs = load_all_img(path)
    for img,filename in imgs:
        plt.imshow(img, cmap=plt.cm.binary)
        plt.show()
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_NEAREST)
        plt.imshow(img, cmap=plt.cm.binary)
        plt.show()   
        reshaped_imgs.append((img,filename))
    return reshaped_imgs
        

def load_all_img(path='photos'):
    #this actually pulls the images into the array for reshapeAllImg
    
    images = []
    for filename in os.listdir(path):
        
        #print(filename)
        img = cv2.imread(os.path.join(path,filename))
        filename = filename.split(' ',1)[0]
        COLORED = False
        if img.shape[2] > 2:
            COLORED = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.cvtColor(img, cv.CV_GRAY2BGR)
        if img is not None:
            
            #img = np.invert(np.array([img]))
            
            images.append((img,filename))
    return images

def readImg(path):
    #given the img file name itll go make the full path for the reshape function to find it
    directory = os.getcwd()
    directory = directory.replace("\\" ,'/')
    path = directory + '/' + path
    return(path)


def findVal(ar):
    #this function is to find top val's of an array while maintaining their inital index
    #this also will return the second largest index if its within 200 of the largest val.
    for i in range(len(ar)):
        ar[i] = (ar[i],i)
    ar.sort(reverse=True)
    if ar[0][0] > int(ar[1][0]) + 200:
        return ar[0][1], None
    else:
        return ar[0][1], ar[1][1]





if __name__ == '__main__':
    path = 'test.jpg'
    #print(readImg(path))
    
    x = load_all_img()
    
    
    
    
    
    
    
    