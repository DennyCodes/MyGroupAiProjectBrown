from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import random
import pandas as pd
import librosa
import gdown
import soundfile as sf
from scipy.signal import butter, filtfilt
import numpy as np

app = Flask(__name__)
CORS(app)

def resample_audio(path, sampling_rate=500):
    y, sr = librosa.load(path, sr=sampling_rate)
    y, _ = librosa.effects.trim(y)
    if sr != sampling_rate:
        librosa.resample(y, sr, sampling_rate)
    return y  

def std_len(data, target_length):
    print(f"Model input shape: {data.shape}")
    if len(data) < target_length:
        data = np.pad(data, (0, target_length - len(data)))
    else:
        data = data[:target_length]
    print(f"Model output shape: {data.shape}")
    return data

def lowpass(data, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)

    print(f"Filtered audio shape: {filtered_data.shape}")

    return filtered_data


# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/process_audio', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print(request.files)
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        final_audio = resample_audio("/Users/gabrielalvesiervolino/Desktop/Coding/machineLearning/voice_recog_final_project/site/api/uploads/audio.wav", 16000)
        final_audio = lowpass(final_audio, cutoff_freq= 7999, sample_rate=16000) 
        final_audio = std_len(final_audio, 80000)
        sf.write("/Users/gabrielalvesiervolino/Desktop/Coding/machineLearning/voice_recog_final_project/site/api/uploads/audio.wav", final_audio, samplerate=16000)

        y, sr = librosa.load("/Users/gabrielalvesiervolino/Desktop/Coding/machineLearning/voice_recog_final_project/site/api/uploads/audio.wav")
        normalized = librosa.util.normalize(y)

        stft = librosa.core.stft(normalized)
        stft_db = librosa.amplitude_to_db(abs(stft), ref=np.max)

        stft_db = stft_db.flatten()
        stft_db = stft_db[0:100001].astype(int)
        
        random_num = random.randint(0, 10)
        return jsonify({'message': "gabe", 'file_path': file_path}), 200

if __name__ == '__main__':
    app.run(debug=True)

#curl -X POST -F "file=@file_dir" http://127.0.0.1:5000/process_audio
