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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import gc


app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device('cpu')

# Dataset 
class AudioDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        features = self.X[index]
        target = self.y[index]
        return features, target

# DataLoader
def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):

    if test:
        pass
    
    df = pd.read_pickle('final_data.pkl')
    
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

    train_dataset = AudioDataset(X, y)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
 
    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)

# CIFAR10 dataset 
batch_size = 16

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

num_classes = 4
num_epochs = 20
learning_rate = 0.01

model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9)  

best_accuracy = 0.0
best_model_wts = model.state_dict()




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


def predict_single_sample(model, sample):
    print("sample: ")
    print(sample)
    print(sample.shape)
    model.eval()
    with torch.no_grad():
        prepped_sample = sample.unsqueeze(0)
        prepped_sample = prepped_sample.permute(0, 2, 1)

        sample_tensor = prepped_sample.to(device)
        output = model(sample_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()

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

        print(stft_db)
        stft_db = stft_db.flatten()
        stft_db = stft_db[0:100001].astype(int)
        stft_db = torch.tensor(stft_db, dtype=torch.float32).unsqueeze(1)
        print(stft_db.shape)
        

        # Load best model weights
        model = ResNet(ResidualBlock, [3, 4, 6, 3], 4).to(device)
        model.load_state_dict(torch.load("/Users/gabrielalvesiervolino/Desktop/Coding/machineLearning/voice_recog_final_project/best_model.pth", map_location=torch.device('cpu')))
        
        output = predict_single_sample(model, stft_db)
        print("output: ")
        print(output) 
        output_dict = {0:"murad", 1:"gabe", 2:"deniz", 3:"bora"}


        print(f'Best validation accuracy: {best_accuracy} %')
        return jsonify({'message': str(output_dict[output]), 'file_path': file_path}), 200

if __name__ == '__main__':
    app.run(debug=True)




#curl -X POST -F "file=@file_dir" http://127.0.0.1:5000/process_audio
