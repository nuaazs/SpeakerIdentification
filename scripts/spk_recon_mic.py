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
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import SpeakerRecognition
import torchaudio
import random
from tqdm import tqdm
import pickle
import time
from multiprocessing.dummy import Pool as ThreadPool
import pickle

raw_path = "/mnt/malixin/malixin/afterpro_mic_formal_8k"


cn_wavs = getCNCeleb(raw_path,1500)
cn_wavs_dict = getCNCeleb_dict(raw_path,1500)





for time_span in [15,18,21,24]: # ,15,9,12,15,12,15,18,21,24
    used = []
    pass_count = 0
    spkreg = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./pretrained_ecapa")
    database = {}
    testbase = {}
    spk_num = 0
    pbar = tqdm(cn_wavs_dict.keys())
    for spk in pbar:
        if spk not in database.keys():
            for wav_file in cn_wavs_dict[spk]:
                file_name = wav_file.split("/")[-1]
                wav, sr = sf.read(wav_file)
                #print(sr)
                #print(int(len(wav)/sr))
                length = int(len(wav)/2)
                time_length = int(length/sr)
                #print(time_length)
                wav_1 = torch.tensor(wav[:length]).unsqueeze(0)
                wav_2 = torch.tensor(wav[length:]).unsqueeze(0)
                if time_span<= time_length:
                    embedding_1 = spkreg.encode_batch(wav_1[0][:sr*time_span])
                    embedding_2 = spkreg.encode_batch(wav_2[0][:sr*time_span])
                else:
                    pass_count+=1
                    continue
                database[spk] = {"spk_name":spk,"file_name":file_name,"embedding_1":embedding_1,"embedding_2":embedding_2}   
                used.append(wav_file)
                spk_num+=1
                break
            pbar.set_description(f"TimeSpan:{time_span} Save:{spk_num} Pass:{pass_count}")
            
    f_save = open(f"mic_database_1580_8000Hz_{time_span}s.pkl", 'wb')
    pickle.dump(database, f_save)
    f_save.close()
    print(f"time_span:{time_span}:{spk_num}")
    

similarity = torch.nn.CosineSimilarity(dim=-1,eps=1e-6)

def sort_result(x):
    return float(x[0])*(-1)

for time_span in [12,15,18,21,24]:# ,9,12,15
    total_test_num = 0
    top_num = 10
    top=np.array([0]*top_num)

    # read database
    f_read = open(f"mic_database_1580_8000Hz_{time_span}s.pkl", 'rb')
    database = pickle.load(f_read)
    f_read.close()
    
    print(f"Time Span: {time_span}\nSpk Num: {len(database.keys())}")
    
    for item in database.keys():
        total_test_num += 1
        results = []
        embedding_1 = torch.tensor(database[item]["embedding_1"])
        embedding_2 = torch.tensor(database[item]["embedding_2"])
        #print(len(embedding_1))
        name = database[item]["spk_name"]
        #print(f"Self test:{name}->   {similarity(embedding_1, embedding_2)}")
        for base_item in database:
            base_embedding_1 = torch.tensor(database[base_item]["embedding_1"])
            results.append([similarity(embedding_2, base_embedding_1),base_item])
        results = sorted(results,key=sort_result)

        for index in range(5):
            result = results[index]
            #print(result)
            if item in result:
                for ii in range(index,top_num):
                    top[ii]+=1
    print(top/total_test_num * 100)