import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
# 加载音频文件
y, sr = librosa.load("C:/Users/小山/Desktop/Sound/1694668365-4.wav")

# 绘制波形图
plt.figure()
librosa.display.waveshow(y, sr=sr)  # 注意: 这里传递的是音频信号数组y和采样率sr
plt.title('Waveform')
plt.show()


# 計算短時距傅立葉變換
S = np.abs(librosa.stft(y))

# 繪製短時距傅立葉變換圖
fig, ax = plt.subplots()
img = librosa.display.specshow(
    librosa.amplitude_to_db(S, ref=np.max),
    y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()


# 計算梅爾頻率倒譜係數
mfccs = librosa.feature.mfcc(y=y, sr=sr)

# 繪製梅爾頻率倒譜係數圖
fig, ax = plt.subplots()
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='MFCC')
plt.show()