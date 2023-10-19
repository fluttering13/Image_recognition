import torchaudio
import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
from IPython.display import Audio
from speechbrain.pretrained import WaveformEnhancement
'''
SepFormer trained on Microsoft DNS-4 (Deep Noise Suppression Challenge 4 – ICASSP 2022) for speech enhancement (16k sampling frequency)
'''
path = './YT_analysis/audio/6F25QdYp02w.wav'
save_path='./YT_analysis/audio/6F25QdYp02w_denoise.wav'


enhance_model = WaveformEnhancement.from_hparams(
    source="speechbrain/mtl-mimic-voicebank",
    savedir="pretrained_models/mtl-mimic-voicebank",
)
enhanced = enhance_model.enhance_file(path)

# Saving enhanced signal on disk
torchaudio.save(save_path, enhanced.unsqueeze(0).cpu(), 16000)
# signal = read_audio(path).squeeze()
# Audio(signal, rate=8000)
# Audio(enhanced_speech[:, :].detach().cpu().squeeze(), rate=8000)


