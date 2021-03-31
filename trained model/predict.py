import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt

model=load_model('alzheimers.h5')

img_pred=cv2.imread("test/NonDemented/26.jpg")

predicted_data=np.expand_dims(img_pred,axis=0)
prediction=model.predict(predicted_data)

i=0

for i in range(4):
    if prediction[0][i]==1.00:
        if i==0:
            print('Mild Demented')
        elif i==1:
            print('Modearate Demented')
        elif i==2:
            print('No Alzheimer')
        else:
            print('Very Mild Demented')

    i+=1

plt.imshow(img_pred)
plt.show()
