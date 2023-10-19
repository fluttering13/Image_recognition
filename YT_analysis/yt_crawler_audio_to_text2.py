
import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pickle
import whisper
from whisper.utils import get_writer


code='6F25QdYp02w'

folder_path='./YT_analysis/audio_to_text/'
file_path='./YT_analysis/audio/'+code+'.wav'



model = whisper.load_model("base")
result = model.transcribe(file_path)
fp=open(folder_path+'code'+'_reuslt.pkl', 'wb')
pickle.dump(result, fp)

txt_writer = get_writer("txt", folder_path)
writer_args = {'highlight_words': False, 'max_line_count': None, 'max_line_width': None}
txt_writer(result, code+'_txt', writer_args)

srt_writer = get_writer("srt", folder_path)
srt_writer(result, code+'_src', writer_args)



# load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio(file_path)
# print(audio.shape)
# audio = whisper.pad_or_trim(audio)
# print(audio.shape)

# make log-Mel spectrogram and move to the same device as the model
# print(len(audio))
# audio=
# mel = whisper.log_mel_spectrogram(audio).to(model.device)
# options = whisper.DecodingOptions(language='zh')
# result = whisper.decode(model, mel, options)
# print(result.text)