import shutil
import librosa
import numpy as np
import pickle
import os
import pyaudio

Soundlen_max = 167
modle_path = "./data/model/"
modle_name = 'svm_model'
modle_name = modle_path + modle_name + '.pkl'

with open(modle_name, 'rb') as model_file:
    svm_model = pickle.load(model_file)

def extract_audioEnergy_features_from_data(audio_data):
    audioEnergy = librosa.feature.rms(y=audio_data)
    return audioEnergy

def detect_and_classify_from_data(audio_data, model):
    features = extract_audioEnergy_features_from_data(audio_data)
    num_zero = Soundlen_max - features.shape[1]
    features = np.pad(features, ((0, 0), (0, num_zero)), mode='constant')
    features_reshaped = features.reshape(1, -1)
    prediction = model.predict(features_reshaped)[0]
    return prediction

# 滑动窗口参数
window_size = 1024  # 窗口大小（可以根据需要调整）

# 初始化音频流
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True, frames_per_buffer=window_size)

print("开始实时检测音频...")
while True:
    try:
        audio_data = np.frombuffer(stream.read(window_size), dtype=np.float32)
        prediction = detect_and_classify_from_data(audio_data, svm_model)
        if prediction == 1:
            print("检测到目标声音")
        # else:
            # print("背景声音")
    except KeyboardInterrupt:  # 用户按下Ctrl+C来停止实时检测
        break

# 关闭音频流和PyAudio会话
stream.stop_stream()
stream.close()
p.terminate()
