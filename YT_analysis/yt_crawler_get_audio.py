import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from pytube import YouTube
from moviepy.editor import *
from bs4 import BeautifulSoup
from selenium import webdriver
import time
from selenium.webdriver.common.by import By
import requests
from pydub import AudioSegment     # 載入 pydub 的 AudioSegment 模組
from os import path
import subprocess

def get_url_code(url):
    string=''
    count=0
    for word in url:
        if count>=1:
            string=string+word    
        if word=='=':
            count=count+1
    return string

save_path='./YT_analysis/audio/'
fp=open('./YT_analysis/dict_url.pkl', 'rb')
urls=pickle.load(fp)
url=urls['url_list'][63]
code=get_url_code(url)

yt=YouTube(url)
file_path='./YT_analysis/audio/'+code+'.mp3'
new_file_path='./YT_analysis/audio/'+code+'.wav'
yt.streams.filter().get_audio_only().download(filename=file_path)

# print(os.path.isfile(file_path))
subprocess.call(['ffmpeg', '-i', file_path, new_file_path])
# path=r'C:\Users\Uesr\Desktop\git-folder\Project-recongnition\YT_analysis\audio\6F25QdYp02w.mp3'
# song = AudioSegment.from_file(file_path, format="mp4")    # 讀取 mp3 檔案
# song.export('new_file_path')

# print(AudioSegment.ffmpeg)
# AudioSegment.ffmpeg = os.getcwd()+"\\ffmpeg\\bin\\ffmpeg.exe"
# print (AudioSegment.ffmpeg)