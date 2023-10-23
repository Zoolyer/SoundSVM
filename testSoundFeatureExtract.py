import time
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os
import keyboard
import pickle
import joblib

# 正确声音文件地址C:\Users\小山\Desktop\Sound
rightfiles_path = "C:/Users/小山/Desktop/Sound/"
# 环境声音文件地址
envfiles_path = "C:/Users/小山/Desktop/SoundBG/"
# 模型文件保存地址
modle_path = "D:/TTV-DCMA-09B/data/model/"
# 保存的模型名称（英文）
modle_name = 'svm_model'
# 最大声音长度（视频声音越长该值越大，若检测音频时间长于训练音频则增大该值）
Soundlen_max = 0

################## 需要修改的部分都在上面 ##################


# 提取声音能量特征定义函数
def extract_audioEnergy_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    # audioEnergy = librosa.feature.rms(y=y, frame_length=13)
    # audioEnergy = librosa.feature.rms(y=y, pad_mode='empty')
    audioEnergy = librosa.feature.rms(y=y)
    return audioEnergy


# 模型名称编辑
modle_name = modle_path + modle_name + '.pkl'

# 正确声音文件数量
rightfiles_numbel = 0
audio_rightfiles = []  # 正确声音文件列表
for rightfile_path in os.listdir(rightfiles_path):
    rightfile_path = os.path.join(rightfiles_path, rightfile_path)
    audio_rightfiles.append(rightfile_path)
    rightfiles_numbel += 1




# 环境声音文件数量
envfiles_numbel = 0
audio_envfiles = []  # 环境声音文件列表
for envfile_path in os.listdir(envfiles_path):
    envfile_path = os.path.join(envfiles_path, envfile_path)
    audio_envfiles.append(envfile_path)
    envfiles_numbel += 1


# 标签集
labels_right = np.ones(rightfiles_numbel)
labels_env = np.zeros(envfiles_numbel)
labels = np.hstack((labels_right, labels_env))

# 声音列表
X = []
# 初始正确声音列表
Rights_0 = [extract_audioEnergy_features(audio_rightfile) for audio_rightfile in audio_rightfiles]
# 初始环境声音列表
Envs_0 = [extract_audioEnergy_features(audio_envfile) for audio_envfile in audio_envfiles]



# 查找最大声音长度
for Right_0 in Rights_0:
    if Soundlen_max < Right_0.shape[1]:
        Soundlen_max = Right_0.shape[1]


for Env_0 in Envs_0:
    if Soundlen_max < Env_0.shape[1]:
        Soundlen_max = Env_0.shape[1]



# 正确声音入集
for Right_0 in Rights_0:
    num_zero = Soundlen_max - Right_0.shape[1]

    Right_0 = np.pad(Right_0, ((0, 0), (0, num_zero)), mode='constant')
    X.append(Right_0)




# 环境声音入集
for Env_0 in Envs_0:
    num_zero = Soundlen_max - Env_0.shape[1]
    Env_0 = np.pad(Env_0, ((0, 0), (0, num_zero)), mode='constant')
    X.append(Env_0)

# 构建特征矩阵 X 和标签向量 y
X = np.array(X)

y = np.array(labels)
# 特征矩阵处理
X = np.squeeze(X)


print(X.shape)
# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# 模型初始化，使用高斯核函数
svm_model = SVC(kernel='rbf')
# 训练模型
svm_model.fit(X_train, y_train)

# 模型预测
y_pred = svm_model.predict(X_test)
# 查看模型准确率
# accuracy = accuracy_score(y_test, y_pred)
# print(f"模型准确率：{accuracy}")
classification = classification_report(y_test, y_pred)
print(y_test.shape)
print(y_pred.shape)
print(f"模型分类报告率：{classification}")


with open(modle_name, 'wb') as model_file:
    pickle.dump(svm_model, model_file)