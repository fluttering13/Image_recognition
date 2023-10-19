from pytube import YouTube
from moviepy.editor import *

# def get_audio_from_video(video_path): 
#     """
#     从视频中提取音频
#     :param video:
#     :return:
#     """ 
#     file_path = './source/' + str(np.random.randint(1000000)) + '.wav' 
#     video = VideoFileClip(video_path)
#     audio = video.audio
#     audio.write_audiofile(file_path)
#     return file_path

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

target_url = "https://www.youtube.com/watch?v=4lk7o20e34c"
yt = YouTube(target_url)
print(yt.title)           # 影片標題
print(yt.length)          # 影片長度 ( 秒 )
print(yt.author)          # 影片作者
print(yt.channel_url)     # 影片作者頻道網址
print(yt.thumbnail_url)   # 影片縮圖網址
print(yt.views)           # 影片觀看數

yt_length=yt.length
channel_url=yt.channel_url
url_str=''
count=0
for word in channel_url:
    if count>=4:
        url_str=url_str+word    
    if word=='/':
        count=count+1


path='./YT_analysis/vedio_tmp/'
all_sub_path=[]
for i in range(3):
    all_sub_path.append(path+url_str+str(i)+'.mp4')

for i in range(len(all_sub_path)-1):
    if i==0:
        clip_video1 = VideoFileClip(all_sub_path[i])
        clip_video2 = VideoFileClip(all_sub_path[i+1])
        vedio=transitions_animation(clip_video1, clip_video2)
    else:
        clip_video2 = VideoFileClip(all_sub_path[i+1])
        vedio=transitions_animation(vedio, clip_video2)

vedio.write_videofile(path+url_str+'_final_edit'+'.mp4',temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")