import os
from keras.backend import argmax
import pandas as pd
from pickle import load
from posixpath import dirname
import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import h5py
import librosa
import librosa.display

app = Flask(__name__)
model = load_model('model.hdf5')

p = "/home/abhinavgorantla/hdd/hackathons/dog_cat"


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None 
     
    return mfccsscaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    file = request.files['file']
    if file:
        filename = 'audio.wav'
        file.save(os.path.join(p, filename))
    file_name = "audio.wav"
    
    features = []
    data = extract_features(file_name)
    features.append([data])

    n_rows=40
    n_cols=1
    n_channels=1
    
    df_features=pd.DataFrame(features,columns=['feature'])
    
    X = np.array(df_features.feature.tolist())
    X=X.reshape(X.shape[0],n_rows,n_cols,n_channels)

    prediction = model.predict(X)

    if(argmax(prediction[0]) == 0):
        pred = "cat"
    else:
        pred = "dog"    

    return render_template('index.html', prediction_text='Voice is from a {}.'.format(pred))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
