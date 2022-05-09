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

def testWav(wav,spkreg,sr=16000,split_num=4,min_length=5,similarity_limit=0.7):
    """
    质量检测
    """
    embedding_list = []
    wav_list = []
    similarity=torch.nn.CosineSimilarity(dim=-1,eps=1e-6)
    
    if len(wav)/sr <= split_num*min_length:
        return False
    length = int(len(wav)/split_num)
    
    for index in range(split_num):
        tiny_wav = torch.tensor(wav[index*length:(index+1)*length]).unsqueeze(0)
        wav_list.append(tiny_wav)
        embedding_list.append(spkreg.encode_batch(tiny_wav)[0][0])    
    for embedding1 in embedding_list:
        for embedding2 in embedding_list:
            score = similarity(embedding1, embedding2)
            if score<similarity_limit:
                print(f"Score:{score}")
                return False
    return True



def make_data_base(item):
    spk,wav_filepaths_list = item
    # time_span = 10
    save_path = '/mnt/cjth_database_0508_16k_0.7'
    try:
        
        
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
            print(f"sr:{sr}")
            if not testWav(wav,spkreg):
                print("wav==1 ,不达标")
                return
            print("wav==1 ,达标")
            length = int(len(wav)/2)

            wav_1 = torch.tensor(wav[:length])
            wav_2 = torch.tensor(wav[length:])

            wav_1 = wav_1
            wav_2 = wav_2
            # seg_length = sr*time_span
            
            # if len(wav_2) <= seg_length*4:
            
            #     return
            # wav1_seg_1 = wav_1[seg_length*0:seg_length*1]
            # wav1_seg_2 = wav_1[seg_length*1:seg_length*2]
            # wav1_seg_3 = wav_1[seg_length*2:seg_length*3]
            # wav1_seg_4 = wav_1[seg_length*3:seg_length*4]
            
            # wav2_seg_1 = wav_2[seg_length*0:seg_length*1]
            # wav2_seg_2 = wav_2[seg_length*1:seg_length*2]
            # wav2_seg_3 = wav_2[seg_length*2:seg_length*3]
            # wav2_seg_4 = wav_2[seg_length*3:seg_length*4]

            #cprint(f"{spkreg.encode_batch(wav).numpy()[0][0].shape} reuls")
            # embedding_list_1  = np.array([spkreg.encode_batch(wav_1).numpy()[0][0],
            #                               spkreg.encode_batch(wav1_seg_1).numpy()[0][0],
            #                               spkreg.encode_batch(wav1_seg_2).numpy()[0][0],
            #                               spkreg.encode_batch(wav1_seg_3).numpy()[0][0],
            #                               spkreg.encode_batch(wav1_seg_4).numpy()[0][0]
            #                              ])
            # embedding_list_2  = np.array([spkreg.encode_batch(wav_2).numpy()[0][0],
            #                               spkreg.encode_batch(wav2_seg_1).numpy()[0][0],
            #                               spkreg.encode_batch(wav2_seg_2).numpy()[0][0],
            #                               spkreg.encode_batch(wav2_seg_3).numpy()[0][0],
            #                               spkreg.encode_batch(wav2_seg_4).numpy()[0][0]
            #                              ])
            np.save(os.path.join(save_path_1,f"{spk}.npy"),spkreg.encode_batch(wav_1).numpy()[0][0])
            np.save(os.path.join(save_path_2,f"{spk}.npy"),spkreg.encode_batch(wav_2).numpy()[0][0])
    #         print(embedding_list_1.shape)
    #         print(embedding_list_2.shape)
            print("Saved!")

        elif len(wav_filepaths_list) > 1:
            embedding = []
            for wav_file in wav_filepaths_list:
                wav, sr = sf.read(wav_file)
                print(f"sr:{sr}")
                if not testWav(wav,spkreg):
                    print("wav>1 ,不达标")
                    continue
                print("wav>1 ,达标")
                # seg_length = sr*time_span
                wav = torch.tensor(wav)

                # if len(wav)<= seg_length*4:
                #     continue
                # wav_seg_1 = wav[seg_length*0:seg_length*1]
                # wav_seg_2 = wav[seg_length*1:seg_length*2]
                # wav_seg_3 = wav[seg_length*2:seg_length*3]
                # wav_seg_4 = wav[seg_length*3:seg_length*4]
                
                # embedding_list  = np.array([spkreg.encode_batch(wav).numpy()[0][0],
                #                             spkreg.encode_batch(wav_seg_1).numpy()[0][0],
                #                             spkreg.encode_batch(wav_seg_2).numpy()[0][0],
                #                             spkreg.encode_batch(wav_seg_3).numpy()[0][0],
                #                             spkreg.encode_batch(wav_seg_4).numpy()[0][0]
                #                            ])
                embedding.append(spkreg.encode_batch(wav).numpy()[0][0])

                if len(embedding) == 2:
                    np.save(os.path.join(save_path_1,f"{spk}_diff.npy"),embedding[0])
                    np.save(os.path.join(save_path_2,f"{spk}_diff.npy"),embedding[1])
                    print("Dif Saved!")
    #                 print(embedding[0].shape)
    #                 print(embedding[1].shape)
                    break
    except Exception as e:
        print(f"** Error->{spk}")
        print(f"** Error->{e}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default="/mnt/cti_record_0508_16k/",help='')
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
