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
from preprocess import audio_change,count_files,data_remove,data_cp,denoise_dict
#import vad
from vad import vad_raw,vad_cat_ndarray,vad_kfth,vad_kfth_multi
from plot import plot_and_play,plot_and_play_list,plot_and_play_dir
from snr import SNR, SNR_power
from get_dataset import getCJKF,getCNCeleb,getVoxCeleb,getCNCeleb_dict,getCJKF_dict
import pickle


raw_path = "/mnt/cti_record_data/"
denoise_path = "/mnt/cti_record_denoise_data"
vad_path = "/mnt/cti_record_vad_data_2/"

## Raw
raw_wavs = getCNCeleb(raw_path,1500)
raw_wavs_dict = getCNCeleb_dict(raw_path,1500)
# plot_and_play(raw_wavs[400],1)

denoise_wavs = getCNCeleb(denoise_path)
denoise_wavs_dict = getCNCeleb_dict(denoise_path)
# plot_and_play_list(denoise_wavs,3,1)

## Vad
from multiprocessing.dummy import Pool as ThreadPool
raw_data_dict = denoise_wavs_dict
new_data_path = vad_path

f_save = open(f"./temp.pkl", 'wb')
pickle.dump(raw_data_dict, f_save)
f_save.close()


def process(item):
    read_vad_save(item,vad_path)
    
    
    
def vad_kfth_multi(raw_data_dict,new_data_path):
    os.makedirs(new_data_path,exist_ok=True)
    
    items = raw_data_dict.keys() # [[key,raw_data_dict,new_data_path] for key in raw_data_dict.keys()]
    #print(items[0][1])
    pool = ThreadPool()
    pool.map(process, items)
    pool.close()
    pool.join()

def read_vad_save(item,new_data_path):
#     raw_data_dict = np.lo
#     # # 读取
    f_read = open("./temp.pkl", 'rb')
    raw_data_dict = pickle.load(f_read)
    f_read.close()

    spk = item
    # print(item)
    wav_file_paths = raw_data_dict[spk]
    for wav_file_path in wav_file_paths:
        wav,sr = sf.read(wav_file_path)
        filename = wav_file_path.split("/")[-1]
        os.makedirs(os.path.join(new_data_path,f"{spk}"),exist_ok=True)
        save_path = os.path.join(new_data_path,f"{spk}/{filename}")
        speech_cat, sr = vad_cat_ndarray(wav[sr*7:,1], sr, save_path=save_path, channel=1)
        
        
vad_kfth_multi(denoise_wavs_dict,vad_path)