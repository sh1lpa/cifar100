import pandas as pd
import sys
import os
from src.exception import CustomException
from src.logger import logging
from tensorflow import keras
import tensorflow as tf
import cv2

class PredictPipeline:
    def __init__(self):
        pass
            
            
    labels = ['aquatic mammals','fish','flowers','food containers','fruit and vegetables','household electrical devices','household furniture','insects','large carnivores','large man-made outdoor things','large natural outdoor scenes','large omnivores and herbivores','medium-sized mammals','non-insect invertebrates','people','reptiles','small mammals','trees','vehicles 1','vehicles']

    
    model = keras.models.load_model('./src/models/cifar20')

    
    def train(self, features):
        
        try:
            print(features)
            image = cv2.imread('./src/uploads/'+features)
            print(image.shape)
            class_name = self.model.predict(cv2.resize(image,(32,32)).reshape(1,32,32,3)).argmax()
            class_name  =  self.labels[class_name]
            return class_name
        except Exception as e:
           raise CustomException(e,sys)
        