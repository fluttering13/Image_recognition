
import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pickle
import whisper
from whisper.utils import get_writer


# code='6F25QdYp02w'

folder_path='./YT_analysis/audio_to_text/'
file_path='./YT_analysis/audio/obqrIjodgWY.wav'
name=''
count=0
dot_count=0
for word in file_path:
    if count==3:
        name=name+word
    if word=='/':
        count=count+1
    if word=='.':
        dot_count=dot_count+1
    if dot_count==2:
        break

model = whisper.load_model("base")
result = model.transcribe(file_path)
fp=open(folder_path+name+'_reuslt.pkl', 'wb')
pickle.dump(result, fp)

txt_writer = get_writer("txt", folder_path)
writer_args = {'highlight_words': False, 'max_line_count': None, 'max_line_width': None}
txt_writer(result, name, writer_args)

srt_writer = get_writer("srt", folder_path)
srt_writer(result, name, writer_args)


