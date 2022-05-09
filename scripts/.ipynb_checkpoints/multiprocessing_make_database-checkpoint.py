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
import torchaudio
import random
from tqdm import tqdm
import pickle
import argparse
from multiprocessing.dummy import Pool as ThreadPool
sys.path.append('/mnt/zhaosheng/brain/utils')

# SpeechBrain
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import SpeakerRecognition

# Utils
from preprocess import audio_change,count_files,data_remove,data_cp,denoise_dict
from vad import vad_raw,vad_cat_ndarray,vad_kfth,vad_kfth_multi
from plot import plot_and_play,plot_and_play_list,plot_and_play_dir
from snr import SNR, SNR_power
from get_dataset import getCJKF,getCNCeleb,getVoxCeleb,getCNCeleb_dict,getCJKF_dict

# Plt Style
plt.rcParams['figure.figsize'] = (20,3)
plt.style.use('ggplot')

def testWav(wav,spkreg,sr=8000,split_num=4,min_length=5,similarity_limit=0.0):
    """
    质量检测
    """
    embedding_list = []
    wav_list = []
    similarity=torch.nn.CosineSimilarity(dim=-1,eps=1e-6)
    
    if len(wav)/sr <= split_num*min_length:
        return False
    length = int(len(wav)/4)
    
    for index in range(split_num):
        tiny_wav = torch.tensor(wav[index*length:(index+1)*length]).unsqueeze(0)
        wav_list.append(tiny_wav)
        embedding_list.append(spkreg.encode_batch(tiny_wav)[0][0])    
    for embedding1 in embedding_list:
        for embedding2 in embedding_list:
            if similarity(embedding1, embedding2)<similarity_limit:
                return False
    return True



def make_data_base(item):
    spk,wav_filepaths_list = item
    time_span = 5
    save_path = '/mnt/cjth_database_voxceleb_0424_0.8'
    
#     spkreg = EncoderClassifier.from_hparams(
#         source='/mnt/zhaosheng/brain/results/voxceleb12_ecapa_augment/22041901/save/CKPT+2022-04-19+14-43-59+00',
#         hparams_file='/mnt/zhaosheng/brain/yamls/val_hyperparams.yaml')
    spkreg = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./pretrained_ecapa")
    # "speechbrain/spkrec-ecapa-voxceleb"   "LanceaKing/spkrec-ecapa-cnceleb"

    save_path_1 = os.path.join(save_path,"embedding_1")
    save_path_2 = os.path.join(save_path,"embedding_2")
    os.makedirs(save_path_1,exist_ok=True)
    os.makedirs(save_path_2,exist_ok=True)
    basedone=False
    
    
    if len(wav_filepaths_list) < 1:
        return

    elif len(wav_filepaths_list) == 1:
        wav_file = wav_filepaths_list[0]
        wav, sr = sf.read(wav_file)
        if not testWav(wav,spkreg):
            print("测试不达标")
            return
        length = int(len(wav)/2)
        time_length = int(length/sr)
        wav_1 = torch.tensor(wav[:length]).unsqueeze(0)
        wav_2 = torch.tensor(wav[length:]).unsqueeze(0)
        seg_length = sr*time_span
        print(wav_2.shape)
        print(wav_2[0].shape)
        if len(wav_2[0]) <= seg_length*4:
        
            return
        wav1_seg_1 = wav_1[:,seg_length*0:seg_length*1]
        wav1_seg_2 = wav_1[:,seg_length*1:seg_length*2]
        wav1_seg_3 = wav_1[:,seg_length*2:seg_length*3]
        wav1_seg_4 = wav_1[:,seg_length*3:seg_length*4]
        
        wav2_seg_1 = wav_2[:,seg_length*0:seg_length*1]
        wav2_seg_2 = wav_2[:,seg_length*1:seg_length*2]
        wav2_seg_3 = wav_2[:,seg_length*2:seg_length*3]
        wav2_seg_4 = wav_2[:,seg_length*3:seg_length*4]
        embedding_list_1  = np.array([spkreg.encode_batch(wav_1).numpy()[0][0],
                                      spkreg.encode_batch(wav1_seg_1).numpy()[0][0],
                                      spkreg.encode_batch(wav1_seg_2).numpy()[0][0],
                                      spkreg.encode_batch(wav1_seg_3).numpy()[0][0],
                                      spkreg.encode_batch(wav1_seg_4).numpy()[0][0]
                                     ])
        embedding_list_2  = np.array([spkreg.encode_batch(wav_2).numpy()[0][0],
                                      spkreg.encode_batch(wav2_seg_1).numpy()[0][0],
                                      spkreg.encode_batch(wav2_seg_2).numpy()[0][0],
                                      spkreg.encode_batch(wav2_seg_3).numpy()[0][0],
                                      spkreg.encode_batch(wav2_seg_4).numpy()[0][0]
                                     ])
        np.save(os.path.join(save_path_1,f"{spk}.npy"),embedding_list_1)
        np.save(os.path.join(save_path_2,f"{spk}.npy"),embedding_list_2)
#         print(embedding_list_1.shape)
#         print(embedding_list_2.shape)
        print("Saved!")

    elif len(wav_filepaths_list) > 1:
        embedding = []
        for wav_file in wav_filepaths_list:
            wav, sr = sf.read(wav_file)
            if not testWav(wav,spkreg):

                continue
                
            seg_length = sr*time_span
            wav = torch.tensor(wav).unsqueeze(0)

            if len(wav[0])<= seg_length*4:
                continue
            wav_seg_1 = wav[:,seg_length*0:seg_length*1]
            wav_seg_2 = wav[:,seg_length*1:seg_length*2]
            wav_seg_3 = wav[:,seg_length*2:seg_length*3]
            wav_seg_4 = wav[:,seg_length*3:seg_length*4]
            embedding_list  = np.array([spkreg.encode_batch(wav).numpy()[0][0],
                                        spkreg.encode_batch(wav_seg_1).numpy()[0][0],
                                        spkreg.encode_batch(wav_seg_2).numpy()[0][0],
                                        spkreg.encode_batch(wav_seg_3).numpy()[0][0],
                                        spkreg.encode_batch(wav_seg_4).numpy()[0][0]
                                       ])
            embedding.append(embedding_list)

            if len(embedding) == 2:
                np.save(os.path.join(save_path_1,f"{spk}_diff.npy"),embedding[0])
                np.save(os.path.join(save_path_2,f"{spk}_diff.npy"),embedding[1])
                print("Dif Saved!")
#                 print(embedding[0].shape)
#                 print(embedding[1].shape)
                break

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default="/mnt/cti_record_0424/wav",help='')
    args = parser.parse_args()

    similarity = torch.nn.CosineSimilarity(dim=-1,eps=1e-6)
    vad_path = args.dataset
    vad_wavs = getCNCeleb(vad_path)
    vad_wavs_dict = getCNCeleb_dict(vad_path)

    items = [[key,vad_wavs_dict[key]] for key in vad_wavs_dict.keys()]
#     print(items[0])
#     for item in items[:100]:
#         print(item)
#         make_data_base(item)
    pool = ThreadPool()
    pool.map(make_data_base, items)
    pool.close()
    pool.join()
