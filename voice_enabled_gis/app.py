
from flask import Flask, render_template, request, jsonify
import torch
import speech_recognition as sr
import numpy as np
import librosa
import requests
from torch import nn

app = Flask(__name__)

class VoiceEmotionNN(nn.Module):
    def __init__(self):
        super(VoiceEmotionNN, self).__init__()
        self.fc1 = nn.Linear(40, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = VoiceEmotionNN()
model.load_state_dict(torch.load('model/voice_emotion_model.pth'))
model.eval()

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/voice-command', methods=['POST'])
def voice_command():
    recognizer = sr.Recognizer()
    audio_file = request.files['audio']
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        command = recognizer.recognize_google(audio)
        audio_features = extract_features(audio_file)
        audio_features = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0)
        prediction = model(audio_features)
        emotion = torch.argmax(prediction, dim=1).item()
        return jsonify({'command': command, 'emotion': emotion})
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio'})
    except sr.RequestError as e:
        return jsonify({'error': f'Request error: {e}'})

@app.route('/get-osm-data', methods=['GET'])
def get_osm_data():
    query = """
    [out:json][timeout:25];
    (
      way["highway"](around:1000,47.4979,19.0402);
    );
    out body;
    >;
    out skel qt;
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={'data': query})
    return response.json()

if __name__ == '__main__':
    app.run(debug=True)
