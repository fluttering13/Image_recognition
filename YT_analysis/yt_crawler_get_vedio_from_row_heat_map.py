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

def transitions_animation(clip_video1, clip_video2): 
    """ 
    兩段視訊中轉場動畫（以淡入淡出為例） 
    注意：保證視訊拍攝幀率一致 
    :param video1: 
    :param video2: 
    :return: 
    """ 
    # 獲取視訊時長 

    duration_video1 = clip_video1.duration 

    duration_video2 = clip_video2.duration 

    # 獲取視訊音訊 

 
    audio_video1 = clip_video1.audio
    audio_video2 = clip_video2.audio
 

 
    print(f'兩段視訊的時長分別為:{duration_video1},{duration_video2}') 
 
    # 統一視訊解析度 
    w, h, fps = clip_video1.w, clip_video1.h, clip_video1.fps 
    clip_video2_new = clip_video2.resize((w, h)) 
 
    # 轉場時長，預設2s 
    transitions_time = 2 
 
    # 第一段視訊執行淡出效果 
    subVideo1_part1 = clip_video1.subclip(0, duration_video1 - 2) 
    subVideo1_part2 = clip_video1.subclip(duration_video1 - 2).fadeout(2, (1, 1, 1)) 
 
    # 第二段視訊執行淡入效果 
    subVideo2_part1 = clip_video2_new.subclip(0, 3).fadein(3, (1, 1, 1)) 
    subVideo2_part2 = clip_video2_new.subclip(3) 
 
    # 合併4段視訊 
    result_video = concatenate_videoclips([subVideo1_part1, subVideo1_part2, subVideo2_part1, subVideo2_part2]) 
 
    # 合併音訊 
    result_audio = concatenate_audioclips([audio_video1, audio_video2]) 
 
    # 視訊設定音訊檔案 
    final_clip = result_video.set_audio(result_audio)
    return final_clip

target_url = "https://www.youtube.com/watch?v=obqrIjodgWY"
# content=requests.get(target_url).text
# soup=BeautifulSoup(content,'html.parser')
# obj=soup.findAll('path',attrs={'class','ytp-heat-map-path'})
# print(obj)


options = webdriver.ChromeOptions()
options.add_argument("headless")
options.add_argument('blink-settings=imagesEnabled=false') 
options.add_argument('--no-sandbox')
driver = webdriver.Chrome(options=options)
driver.get(target_url)
driver.maximize_window()
time.sleep(5)
heat_map_path=driver.find_element(By.CLASS_NAME,'ytp-heat-map-path')
print(heat_map_path)
str_heat_map_path=heat_map_path.get_attribute('d')
heat_map_row_data={'heat_map_row_data':str_heat_map_path}
fp=open('./YT_analysis/heat_map_row_data.pkl', 'wb')
pickle.dump(heat_map_row_data,fp)
# ###load data
fp=open('./YT_analysis/heat_map_row_data.pkl', 'rb')
row_data=pickle.load(fp)
row_data=row_data['heat_map_row_data']
# print(row_data)
find_all_list=re.findall('C.*?C',row_data)
x_list=[]
y_list=[]
for one_list in find_all_list:
    string=one_list[2:-2]
    split_list=re.split(' ',string)
    for one_split_list in split_list:
        x_y_list=re.split(',',one_split_list)
        x_list.append(float(x_y_list[0]))
        y_list.append(100-float(x_y_list[1]))

y_hat_list=savgol_filter(y_list, 5, 1)
plt.plot(x_list,y_list)
plt.plot(x_list,y_hat_list, color='red')
plt.show()
print(y_hat_list)

y_list=y_hat_list
number_of_max=3
print(len(x_list))
print(len(y_list))
intervel=int(len(x_list)/3)

time_split_list=[]
for i in range(number_of_max):
    #print(i)
    tmp_y_list=y_list[0+i*intervel:(i+1)*intervel]
    # print(0+i*intervel,(i+1)*intervel)
    left_index=np.argmax(tmp_y_list)+i*intervel
    #print(left_index)
    right_index=left_index.copy()
    #往左走
    if left_index==i*intervel:
        pass
    else:
        while y_list[left_index]>y_list[left_index-1]:
            if left_index==(i)*intervel:
                #print('l touch the bound')
                break
            left_index=left_index-1
        #print('now right index',right_index,left_index)
        #print((i+1)*intervel)
    if right_index==(i+1)*intervel-1:
        pass
    else:
        while y_list[right_index]>y_list[right_index+1]:
            #print(right_index,(i+1)*intervel)
            if right_index==(i+1)*intervel:
                #print('r touch the bound')
                break    
            right_index=right_index+1
    time_split_list.append([left_index,right_index])
    # print(x_list[left_index],x_list[right_index])

yt = YouTube(target_url)
print(yt.title)           # 影片標題
print(yt.length)          # 影片長度 ( 秒 )
print(yt.author)          # 影片作者
print(yt.channel_url)     # 影片作者頻道網址
print(yt.thumbnail_url)   # 影片縮圖網址
print(yt.views)           # 影片觀看數

yt_length=yt.length

url_str=''
count=0
for word in target_url:
    if count==1:
        url_str=url_str+word    
    if word=='=':
        count=1


path='./YT_analysis/vedio_tmp/'
###下載原檔
yt.streams.filter().get_highest_resolution().download(filename=path+url_str+'.mp4')
video = VideoFileClip(path+url_str+'.mp4')


all_sub_path=[]
###根據時間條分割影片
for i in range(number_of_max):
    start_time=time_split_list[i][0]/len(x_list)*yt_length
    end_time=time_split_list[i][1]/len(x_list)*yt_length
    output = video.subclip(start_time,end_time)
    output.write_videofile(path+url_str+str(i)+'.mp4',temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    all_sub_path.append(path+url_str+str(i)+'.mp4')

###整合分割影片加轉場特效
for i in range(len(all_sub_path)-1):
    if i==0:
        clip_video1 = VideoFileClip(all_sub_path[i])
        clip_video2 = VideoFileClip(all_sub_path[i+1])
        vedio=transitions_animation(clip_video1, clip_video2)
    else:
        clip_video2 = VideoFileClip(all_sub_path[i+1])
        vedio=transitions_animation(vedio, clip_video2)

vedio.write_videofile(path+url_str+'_final_edit'+'.mp4',temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")