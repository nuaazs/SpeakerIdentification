# 加载需要的依赖
import os
import pydub
from pydub import AudioSegment
import soundfile as sf
import wave
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display,Audio
import torch
import shutil

# VAD工具包相关
import glob
import torch
torch.set_num_threads(1)

from IPython.display import Audio
from pprint import pprint

USE_ONNX = False
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=USE_ONNX)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

    # 函数名：SNR
    # 功能
    #   计算音频的信噪比
    # 输入
    #   file_path:需要音频所在路径；channel：选择需要处理的音频通道，默认channel=0
    # 输出
    #   snr：信噪比的数值 



def SNR(file_path, channel=0):

    s = n = 0
    len_seg = len_silent = 0
    wav, sr = sf.read(file_path)
    if wav.ndim > 1:
        wav = wav[:,channel]
    wav_tensor = torch.tensor(wav.astype('float32'))
    #print(len(wav_tensor))
    speech_timestamps = get_speech_ts(wav_tensor, model, num_steps=4)
    #slice_num = 0
    Silent_sound_start = 0
    for segment in speech_timestamps:
        start = segment['start'] 
        end = segment['end']
        #print(start,end) 
        #slice_num += 1
        #slice_name = f'vad_{slice_num}.wav'
        wav_seg = wav[start:end]
        #print(wav_seg)
        wav_seg_energy = sum(wav_seg ** 2)
        wav_silent = wav[Silent_sound_start:start]
        wav_silent_energy = sum(wav_silent ** 2)
        #print(start,end,Silent_sound_start)
        Silent_sound_start = end
        s += wav_seg_energy
        n += wav_silent_energy
        len_seg += len(wav_seg)
        len_silent += len(wav_silent)
        #print(s)
        #print(n)
        #print(len_silent)
        #print(len_seg)
    #print(speech_slice[0])

    if s == 0:
            return None
    else:
        if n == 0:
            n = 1
            print('No noise')
        return int(10 * np.log10(s / n))
        #s = s / len_seg
        #n = n / len_silent


    # 函数名：SNR_power
    # 功能
    #   计算音频的信号能量和噪声能量
    # 输入
    #   file_path:需要音频所在路径；channel：选择需要处理的音频通道，默认channel=0
    # 输出
    #   ps：信号能量的数值
    #   pn：噪音能量的数值

def SNR_power(file_path, channel=0):

    s = n = 0
    len_seg = len_silent = 0
    wav, sr = sf.read(file_path)
    if wav.ndim > 1:
        wav = wav[:,channel]
    wav_tensor = torch.tensor(wav.astype('float32'))
    #print(len(wav_tensor))
    speech_timestamps = get_speech_ts(wav_tensor, model, num_steps=4)
    #slice_num = 0
    Silent_sound_start = 0
    for segment in speech_timestamps:
        start = segment['start'] 
        end = segment['end']
        #print(start,end) 
        #slice_num += 1
        #slice_name = f'vad_{slice_num}.wav'
        wav_seg = wav[start:end]
        #print(wav_seg)
        wav_seg_energy = sum(wav_seg ** 2)
        wav_silent = wav[Silent_sound_start:start]
        wav_silent_energy = sum(wav_silent ** 2)
        #print(start,end,Silent_sound_start)
        Silent_sound_start = end
        s += wav_seg_energy
        n += wav_silent_energy
        len_seg += len(wav_seg)
        len_silent += len(wav_silent)
    
    if s == 0:
        ps = None
    else:
        ps = int(10 * np.log10(s))
    if n == 0:
        pn = None
    else:
        pn = int(10 * np.log10(n))
    
    res = {'Ps':ps,'Pn':pn}
    return res