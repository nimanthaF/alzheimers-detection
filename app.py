from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
import os

import urllib.request
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)

app.secret_key='secret key'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

@app.route('/display/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
    
	return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/')
def model_prediction(img_path):
    model=load_model('./trained_model/alzheimers.h5')
    #img_path="trained_model/test/ModerateDemented/27.jpg"
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
    plt.imshow(img)
    plt.show()

    return output


if __name__=='__main__':
    app.run()