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

    
    def predict(self, features):
        
        try:
            print(features)
            image = cv2.imread('./src/uploads/'+features)
            print(image.shape)
            class_name = self.model.predict(cv2.resize(image,(32,32)).reshape(1,32,32,3)).argmax()
            class_name  =  self.labels[class_name]
            return class_name
        except Exception as e:
           raise CustomException(e,sys)
        

# class CustomData:
    # def __init__(self,
    #              gender: str,
    #              race_ethnicity: str,
    #              parental_level_of_education: str,
    #              lunch: str,
    #              test_preparation_course: str,
    #              reading_score: int,
    #              writing_score: int):
    #     self.gender=gender

    #     self.race_ethnicity=race_ethnicity
        
    #     self.parental_level_of_education=parental_level_of_education
        
    #     self.lunch=lunch
        
    #     self.test_preparation_course=test_preparation_course
        
    #     self.reading_score=reading_score
        
    #     self.writing_score=writing_score

    # def get_data_as_dataframe(self):
    #     try:
    #         custom_data_input_dict = {
    #             "gender":[self.gender],
    #             "race_ethnicity":[self.race_ethnicity],
    #             "parental_level_of_education":[self.parental_level_of_education],
    #             "lunch":self.lunch,
    #             "test_preparation_course":[self.test_preparation_course],
    #             "reading_score":[self.reading_score],
    #             "writing_score":[self.writing_score]
    #         }

    #         return pd.DataFrame(custom_data_input_dict)
    #     except:
    #         pass