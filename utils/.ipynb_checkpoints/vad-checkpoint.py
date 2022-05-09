

'''
    by yelin
    by zhaozifeng
'''

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
import glob
import torch
torch.set_num_threads(1)
from IPython.display import Audio
from pprint import pprint
import numpy as np
import scipy.io.wavfile as wavfile
import numpy
import os.path
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

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


# VAD核心
def vad(file_path, channel=0):
    speech_slice = []
    wav, sr = sf.read(file_path)
    if wav.ndim > 1:
        wav = wav[:,channel]
    wav_tensor = torch.tensor(wav.astype('float32'))
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr,window_size_samples=512)
    for segment in speech_timestamps:
        start = segment['start'] 
        end = segment['end']
        wav_seg = wav[start:end]
        speech_slice.append(wav_seg)
    return speech_slice, speech_timestamps, wav, wav_tensor,sr


# VAD并拼接
def vad_cat(file_path, save_path=None, channel=0):

    # 调用VAD
    speech_slice, speech_timestamps, wav, wav_tensor,sr = vad(file_path, channel)

    # 合并
    total_len = 0
    for slice in speech_slice:
        total_len += len(slice)
    speech_cat = np.zeros((total_len,))
    start_point = 0
    for slice in speech_slice:
        #print(start_point, len(slice))
        speech_cat[start_point:(start_point+len(slice))] = slice
        start_point += len(slice)

    if (save_path is not None) and (speech_cat.shape[0]>0):
        sf.write(save_path, speech_cat, sr)
    else:
        save_path = None

    return speech_cat, sr, save_path


# VAD，np
def vad_ndarray(wav, sr, channel=0):
    speech_slice = []
    #wav, sr = sf.read(file_path)
    if wav.ndim > 1:
        wav = wav[:,channel]
    wav_tensor = torch.tensor(wav.astype('float32'))
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr,window_size_samples=512)
    #slice_num = 0
    for segment in speech_timestamps:
        start = segment['start']
        end = segment['end']
        #print(start,end) 
        #slice_num += 1
        #slice_name = f'vad_{slice_num}.wav'
        wav_seg = wav[start:end]
        speech_slice.append(wav_seg)
    return speech_slice, speech_timestamps, wav, sr


# VAD并拼接，np
def vad_cat_ndarray(wav, sr, save_path=None, channel=0):
    # 调用VAD
    speech_slice, speech_timestamps, wav, sr = vad_ndarray(wav, sr, channel)
    # 合并
    total_len = 0
    for slice in speech_slice:
        total_len += len(slice)
    speech_cat = np.zeros((total_len,))
    start_point = 0
    for slice in speech_slice:
        #print(start_point, len(slice))
        speech_cat[start_point:(start_point+len(slice))] = slice
        start_point += len(slice)

    if save_path is not None:
        sf.write(save_path, speech_cat, sr)

    return speech_cat, sr


# 单个文件VAD与信噪比检测
def vad_raw(file_path, save_path=None, channel=0,time_limit=6,snr_test=True):
    wav,sr = sf.read(file_path)
    assert sr == 8000,print(f"-> {file_path} !!Error SAMPLING_RATE:{sr}")
    SAMPLING_RATE = 8000
    time = float(len(wav)/sr)
    if time<time_limit:
        return {"qualified":False,"error_type":1,"msg":"length error"}

    wav = read_audio(file_path, sampling_rate=SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE,window_size_samples=512)
    s = n = 0
    len_seg = len_silent = 0
    Silent_sound_start = 0
    if snr_test:
        for timestamp in speech_timestamps:
            start = timestamp['start'] 
            end = timestamp['end']
            wav_seg = wav[start:end]

            wav_seg_energy = sum(wav_seg ** 2)
            wav_silent = wav[Silent_sound_start:start]
            wav_silent_energy = sum(wav_silent ** 2)
            Silent_sound_start = end
            s += wav_seg_energy
            n += wav_silent_energy
            len_seg += len(wav_seg)
            len_silent += len(wav_silent)
        if s == 0:
            snr = None
        else:
            if n == 0:
                n = 1
                print('No noise')
            snr = int(10 * np.log10(s / n))
            print(f"s:{s},n:{n},snr:{snr}")
    if len(speech_timestamps)<1:
        return {"qualified":False,"error_type":2,"msg":"vad error"}
    total_time = 0
    for timestamp in speech_timestamps:
        total_time += (timestamp['end']-timestamp['start'])/sr
    if total_time<time_limit:
        return {"qualified":False,"error_type":3,"msg":"vad length error"}
    if save_path:
        save_audio(save_path,
                   collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE)
    return {"qualified":True,"msg":save_path}


# 客服通话数据VAD
def vad_kfth(raw_data_dict,new_data_path):
    os.makedirs(new_data_path,exist_ok=True)
    for spk in tqdm(raw_data_dict.keys()):
        try:
            wav_file_paths = raw_data_dict[spk]
            for wav_file_path in wav_file_paths:
                wav,sr = sf.read(wav_file_path)
                filename = wav_file_path.split("/")[-1]
                os.makedirs(os.path.join(new_data_path,f"{spk}"),exist_ok=True)
                save_path = os.path.join(new_data_path,f"{spk}/{filename}")
                if os.path.exists(save_path):
                    continue
                else:
                    speech_cat, sr = vad_cat_ndarray(wav[sr*7:,1], sr, save_path=save_path, channel=1)
        except:
            print(f"{spk} error")




# 客服通话数据VAD,多线程
def vad_kfth_multi(raw_data_dict,new_data_path):
    os.makedirs(new_data_path,exist_ok=True)
    
    items = [[pbar,key,raw_data_dict[key],new_data_path] for key in raw_data_dict.keys()]
    
    
    pbar = tqdm(total=len(items))

    print(items[0])
    pool = ThreadPool()
    pool.map(read_vad_save, items)
    pool.close()
    pool.join()
    pbar.close()


# vad
def read_vad_save(item):
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=USE_ONNX)
    pbar,spk,wav_file_paths,new_data_path = item
    #print(wav_file_paths)
    for wav_file_path in wav_file_paths:
        
        filename = wav_file_path.split("/")[-1]
        os.makedirs(os.path.join(new_data_path,f"{spk}"),exist_ok=True)
        save_path = os.path.join(new_data_path,f"{spk}/{filename}")
        
        if os.path.exists(save_path):
            continue
        else:
            wav = read_audio(wav_file_path, sampling_rate=8000)
            
            speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=8000,window_size_samples=512)
            
            save_audio(save_path,collect_chunks(speech_timestamps, wav), sampling_rate=8000)
            
            print(save_path)
            #speech_cat, sr = vad_cat_ndarray(wav[sr*7:,1], sr, save_path=save_path, channel=1)
    pbar.update(1)