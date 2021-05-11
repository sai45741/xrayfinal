import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import keras
from keras.models import load_model
import os
import cv2
from werkzeug.utils import secure_filename
#from run import app as application
#import win32api

IMAGE_FOLDER = os.path.join('static', 'images')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER


loaded_model = load_model('cnn_model.h5')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    full_filename = os.path.join(IMAGE_FOLDER,uploaded_file.filename)
    if uploaded_file.filename != '':
        filename=uploaded_file.filename
        uploaded_file.save(full_filename)
    img = cv2.imread(full_filename,cv2.IMREAD_GRAYSCALE)
    img1=cv2.resize(img,(88,88))
    x_in=img1.reshape(88,88,1)
    x_in=np.array([x_in])
    prediction=loaded_model.predict(x_in)

    output = prediction[0][0]
    if output>0.5:
        state='detected'
    else:
        state='not detected'

    return render_template('index.html',image= full_filename,prediction_text='Abnormality {}'.format(state))
    

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
