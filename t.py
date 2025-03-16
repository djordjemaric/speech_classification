import soundfile as sf
import numpy as np
import torch
#nina je najbolja

# data, fs = sf.read("data/SpeechCommands/speech_commands_v0.02/backward/0a2b400e_nohash_0.wav")
data, fs = sf.read("recorded.wav")
tens = torch.Tensor(data)
print(tens.shape[0])
print(fs)
