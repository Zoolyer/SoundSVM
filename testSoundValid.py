import pickle
import librosa
import numpy as np
import pyaudio
import os
import joblib
from sklearn.metrics import classification_report

# 模型路径
modelPath = "D:/TTV-DCMA-09B/data/model/svm_model.pkl"
# 测试音频文件地址
# tests_path = "C:Users/93530/Desktop/SoundBG/1694685135-34.wav"

# 音频特征最大长度
maxFeatureLen = 167


# 提取声音能量特征定义函数
def extract_audioEnergy_features(audio_data, sr):

    # 提取MFCC特征
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)

    # 提取声音能量
    audio_energy = librosa.feature.rms(y=audio_data)

    # 提取色度特征
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, tuning=0.0)

    # 提取过零率
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data)

    # 提取频谱质心
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)

    # 提取频谱带宽
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)

    # print(audio_energy)
    # 连接所有特征成一个单一的矩阵
    all_features = np.concatenate(
        [mfccs, chroma, zero_crossing_rate, spectral_centroid, spectral_bandwidth, audio_energy],
        axis=0  # 使用axis=1将特征连接为一行多列的矩阵
    )
    all_features = all_features.reshape(1,-1)

    return all_features



# 将音频进行分类
def soundSvm(feature):

    if feature.shape[1] > maxFeatureLen:
        feature = feature[:, 0:maxFeatureLen]
    elif feature.shape[1] < maxFeatureLen:

        num_zero = maxFeatureLen - feature.shape[1]
        feature = np.pad(feature, ((0, 0), (0, num_zero)), mode='constant')

    return feature


# 加载模型
with open(modelPath, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# 滑动窗口参数
window_size = 10000  # 窗口大小（可以根据需要调整）

# 初始化音频流
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True, frames_per_buffer=1024)


print("开始检测")
# 实时读取音频流并进行处理
i = 0
while True:
    try:
        audio_data = np.frombuffer(stream.read(window_size), dtype=np.float32)
        # print(audio_data)
        # 提取特征
        audio_feature = extract_audioEnergy_features(audio_data, sr=44100)
        # print(audio_feature)
        audio_feature = soundSvm(audio_feature)

        # 使用加载的模型进行预测
        y_pred = loaded_model.predict(audio_feature)
        print(y_pred)
        if y_pred:
            print(f"{i}   当前有插入检测           ")
            i=i+1

    except KeyboardInterrupt:
        # 用户按下Ctrl+C来停止实时检测
        break

# 关闭音频流和PyAudio会话
stream.stop_stream()
stream.close()
p.terminate()