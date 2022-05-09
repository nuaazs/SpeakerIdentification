#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import thread
import time
from multiprocessing.pool import ThreadPool
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
from tqdm import tqdm,trange
import logging
import pydub
from pydub import AudioSegment
import wave
import pandas as pd
plt.rcParams['figure.figsize'] = (20,3)
sys.path.append('/mnt/zhaosheng/brain/utils')
from preprocess import audio_change,count_files,data_remove,data_cp
#import vad
from vad import vad_raw
from plot import plot_and_play,plot_and_play_list,plot_and_play_dir
from snr import SNR, SNR_power
from get_dataset import getCJKF,getCNCeleb,getVoxCeleb

root_path = "/mnt/zhaozifeng/user_zhaozfieng/"

cn_wavs = getCNCeleb("/mnt/zhaosheng/brain/data/CN-Celeb8k")
# plot_and_play_list(cn_wavs,1)
def plot(filepath):
    print(f"*-> {filepath}")
    wav,sr = sf.read(filepath)
    print('sampling rate =',sr,'Hz')
    display(Audio(data=wav,rate=sr))
    plt.plot(np.arange(len(wav))/sr,wav); plt.xlabel('time / s'); plt.show()

statistics = [0,0,0,0,0]
index = 1
for wav_file in tqdm(cn_wavs):
    spk_name = wav_file.split("/")[-2]
    file_name = wav_file.split("/")[-1]
#     if index == 5:
#         break
    save_path = f"/mnt/cn_data_8k_vad/{spk_name}"
    os.makedirs(save_path,exist_ok=True)
    result = vad_raw(wav_file,os.path.join(save_path,file_name),time_limit=3,snr_test=False)
    if result["qualified"]:
#         print("Before:")
#         plot(wav_file)
#         print("After:")
#         plot(save_path)
        statistics[0]+=1
    else:
        #print(f"Error Type:{result['error_type']}:{result['msg']}")
        statistics[result['error_type']]+=1
    index += 1
print(statistics)



# def vad(item):
#     print(item)
 
# pool_size = 10
# f = open('test.txt', 'r')
# items = f.readlines()
 
# pool = ThreadPool(pool_size)  # 创建一个线程池
# pool.map(my_print, items)  # 往线程池中填线程
# pool.close()  # 关闭线程池，不再接受线程
# pool.join()  # 等待线程池中线程全部执行完