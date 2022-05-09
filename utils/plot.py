import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Audio
import torchaudio
import soundfile as sf
import random
import os

def plot_and_play_dir(path,number,channel=0):
    files = [os.path.join(path,_filename) for _filename in os.listdir(path)]
    for index,file in enumerate(files):
        if index == number:
            break
        wav,sr = sf.read(file)
        if wav.ndim > 1:
            wav = wav[:,channel]
        print('file =',file)
        print('sampling rate =',sr,'Hz')
        display(Audio(data=wav,rate=sr))
        plt.rcParams['figure.figsize'] = (10,2)
        plt.plot(np.arange(len(wav))/sr,wav); plt.xlabel('time / s'); plt.show()
        print("err")
        
def plot_and_play_list(all_wavs,number,channel=0):
    for xx in range(number):
        random_int = random.randint(0,len(all_wavs))
        check_file = all_wavs[random_int]
        try:
            wav,sr = sf.read(check_file)
            if wav.ndim > 1:
                wav = wav[:,channel]
            print('file =',check_file)
            print('sampling rate =',sr,'Hz')
            display(Audio(data=wav,rate=sr))
            plt.rcParams['figure.figsize'] = (10,2)
            plt.plot(np.arange(len(wav))/sr,wav); plt.xlabel('time / s'); plt.show()
        except:
            print("err")

def plot_and_play(check_file,channel=0,title="",showinfo=True):
    try:
        
        wav,sr = sf.read(check_file)
        if wav.ndim > 1:
            wav = wav[:,channel]
        if showinfo:
            print('file =',check_file)
            print('sampling rate =',sr,'Hz')
        display(Audio(data=wav,rate=sr))
        plt.rcParams['figure.figsize'] = (10,2)
        plt.title(title)
        plt.plot(np.arange(len(wav))/sr,wav); plt.xlabel('time / s'); plt.show()
    except:
        print("err")
        
def plot_and_play_array(wav,sr,channel=0,title="",showinfo=True):
    try:
        if wav.ndim > 1:
            wav = wav[:,channel]
        if showinfo:
            print('file =',check_file)
            print('sampling rate =',sr,'Hz')
        display(Audio(data=wav,rate=sr))
        plt.rcParams['figure.figsize'] = (10,2)
        plt.title(title)
        plt.plot(np.arange(len(wav))/sr,wav); plt.xlabel('time / s'); plt.show()
    except:
        print("err")