import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement
###來源自metricgan
path = './YT_analysis/audio/6F25QdYp02w.wav'
save_path='./YT_analysis/audio/6F25QdYp02w_denoise.wav'

enhance_model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus-voicebank",
)

# Load and add fake batch dimension
noisy = enhance_model.load_audio(
    path
).unsqueeze(0)

# Add relative length tensor
enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))

# Saving enhanced signal on disk
torchaudio.save(save_path, enhanced.cpu(), 16000)
print(enhanced)