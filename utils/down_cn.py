import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Audio
import torchaudio
import soundfile as sf
import shutil
import os
import random
from os import listdir, getcwd
import sys
from tqdm import tqdm
import logging

plt.rcParams['figure.figsize'] = (20,3)
sys.path.append('/mnt/zhaosheng/brain/utils')
from preprocess import audio_change,count_files,data_remove,data_cp
import vad

root_path = "/mnt/zhaozifeng/user_zhaozfieng/"


import subprocess
def pre_audio(old_path, new_path, channels, sr, format):
    if os.path.exists(new_path) == False:
        os.mkdir(new_path)
    id_list = os.listdir(old_path)   #  id 列表
    for id in tqdm(id_list): 
        id_path = os.path.join(old_path, id)  # 某个人的路径
        audio_dir = os.listdir(id_path)  #  某个人下的文件夹列表
        new_id_path = os.path.join(new_path, id)
        os.makedirs(new_id_path,exist_ok=True)
        for audio in audio_dir:    # 某个人下的某个文件夹
            audio_path = os.path.join(old_path, id, audio)  #  某个人下的某个文件夹 内的 某条音频 位置
            audio_format = audio_path.split('.')[1]
            old_format = f'.{audio_format}'
            new_name = audio.replace(old_format,format)  # 取新名字
            new_audio_path = os.path.join(new_id_path, new_name)  # 新音频位置
            if os.path.exists(new_audio_path):
                continue
            else:
                cmd = f'ffmpeg -y -i {audio_path} -ac {channels} -ar {sr} {new_audio_path}'
                subprocess.call(cmd, shell=True)

old_path = "/mnt/data_process/CN-Celeb_wav/wav"
new_path = "/mnt/zhaosheng/brain/data/CN-Celeb8k"
channels = 1
sr = 8000
formatf = str('.wav')
pre_audio(old_path, new_path, channels, sr, formatf)
