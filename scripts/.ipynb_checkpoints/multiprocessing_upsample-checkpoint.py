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


sys.path.append('/mnt/zhaosheng/brain/utils')

# SpeechBrain
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import SpeakerRecognition

# Utils
from preprocess import audio_change,count_files,data_remove,data_cp,denoise_dict,audio_change_multi
from vad import vad_raw,vad_cat_ndarray,vad_kfth,vad_kfth_multi
from plot import plot_and_play,plot_and_play_list,plot_and_play_dir
from snr import SNR, SNR_power
from get_dataset import getCJKF,getCNCeleb,getVoxCeleb,getCNCeleb_dict,getCJKF_dict

# Plt Style
plt.rcParams['figure.figsize'] = (20,3)
plt.style.use('ggplot')


raw_path = "/mnt/cti_record_0503/"

## Raw
raw_wavs = getCNCeleb(raw_path)
raw_wavs_dict = getCNCeleb_dict(raw_path,plot=False)

#plot_and_play(raw_wavs[400],1)

## Vad
upsample_path = "/mnt/cti_record_0503_16k/"
audio_change_multi(raw_wavs_dict,upsample_path)

## Denoise
# denoise_dict(raw_wavs_dict,denoise_path)
