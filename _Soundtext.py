import pyaudio
import matplotlib.pyplot as plt
import numpy as np

# 初始化音频流
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,  # 注意: 这里改为paInt16以匹配常见的音频格式
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024)

plt.ion()  # 打开交互模式
fig, ax = plt.subplots()

while True:
    try:
        # 读取音频数据
        input_data = stream.read(1024)
        audio_data = np.frombuffer(input_data, dtype=np.int16)

        # 清除之前的图表数据
        ax.clear()

        # 绘制新的波形图
        ax.plot(audio_data)
        ax.set_title('Real-time Waveform')
        plt.draw()
        plt.pause(0.01)  # 暂停以允许更新

    except KeyboardInterrupt:
        # 如果用户按下Ctrl + C，退出循环
        break

# 停止流并关闭PyAudio
stream.stop_stream()
stream.close()
p.terminate()
