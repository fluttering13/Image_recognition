# Import VoiceFixer!
import numpy as np
from voicefixer import VoiceFixer, Vocoder

path = './YT_analysis/audio/6F25QdYp02w.wav'
save_path='./YT_analysis/audio/6F25QdYp02w_denoise.wav'

path2 = './YT_analysis/audio/6F25QdYp02w2.wav'
save_path2='./YT_analysis/audio/6F25QdYp02w_denoise2.wav'
# Initialize model
voicefixer = VoiceFixer()
# Speech restoration
# Mode 0: Original Model (suggested by default)
# Mode 1: Add preprocessing module (remove higher frequency)
# Mode 2: Train mode (might work sometimes on seriously degraded real speech)
voicefixer.restore(input=path, # input wav file path
                   output=save_path, # output wav file path
                   cuda=False, # whether to use gpu acceleration
                   mode = 0) # You can try out mode 0, 1 to find out the best result

## Initialize a vocoder
# Universal speaker independent vocoder
vocoder = Vocoder(sample_rate=44100) # Only 44100 sampling rate is supported.
### read wave (fpath) -> mel spectrogram -> vocoder -> wave -> save wave (out_path)
# Convert mel spectrogram to waveform
#wave = vocoder.forward(mel=mel_spec) # This forward function is used in the following oracle function.

# Test vocoder using the mel spectrogram of 'fpath', save output to file out_path
vocoder.oracle(fpath=path, # input wav file path
               out_path=save_path2) # output wav file path