# 导入必要的库
import shutil
import librosa
import numpy as np
import pickle
import os

Soundlen_max = 167

modle_path = "./data/model/"
modle_name = 'svm_model'
modle_name = modle_path + modle_name + '.pkl'

rightfiles_path = "./data/Sound/"
envfiles_path = "./data/SoundBG/"

# 如果 ./val/ 文件夹存在，删除它
if os.path.exists('./val/'):
    shutil.rmtree('./val/')

# 创建新的 ./val/ 文件夹和其子文件夹
os.makedirs('./val/Sound/')
os.makedirs('./val/SoundBG/')
# 加载模型



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


with open(modle_name, 'rb') as model_file:
    svm_model = pickle.load(model_file)

def extract_audioEnergy_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    # audioEnergy = librosa.feature.rms(y=y, frame_length=13)
    # audioEnergy = librosa.feature.rms(y=y, pad_mode='empty')
    audioEnergy = librosa.feature.rms(y=y)
    return audioEnergy

# 检测函数
def detect_and_classify(audio_file, model):
    # 提取音频文件的特征
    features = extract_audioEnergy_features(audio_file)

    # 补零以匹配训练数据的长度
    num_zero = Soundlen_max - features.shape[1]
    features = np.pad(features, ((0, 0), (0, num_zero)), mode='constant')

    # 使用模型进行预测之前
    features_reshaped = features.reshape(1, -1)
    prediction = model.predict(features_reshaped)[0]

    return prediction



# 检测并分类
audio_files = audio_rightfiles + audio_envfiles
for audio_file in audio_files:
    prediction = detect_and_classify(audio_file, svm_model)
    # 根据预测结果复制音频文件到相应的分类文件夹
    if prediction == 1:
        shutil.copy(audio_file, './val/Sound/')
    else:
        shutil.copy(audio_file, './val/SoundBG/')