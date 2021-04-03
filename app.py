from flask import Flask, render_template
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
import os

app = Flask(__name__)


@app.route('/')
def model_prediction():
    model=load_model('./trained_model/alzheimers.h5')
    img_path="trained_model/test/ModerateDemented/27.jpg"
    img=image.load_img(img_path,target_size=(208,176))
    predicted_data=np.expand_dims(img,axis=0)
    prediction=model.predict(predicted_data)

    i=0

    for i in range(4):
        if prediction[0][i]==1.00:
            if i==0:
                output='Mild Demented'
            elif i==1:
                output='Moderate Demented'
            elif i==2:
                output='No Alzheimer'
            else:
                output='Very Mild Demented'

        i+=1
    
    return output


if __name__=='__main__':
    app.run()