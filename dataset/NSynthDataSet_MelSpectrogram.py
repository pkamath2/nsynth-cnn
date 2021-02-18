import librosa
import os
import numpy as np
import torch as torch
from torch.utils.data import Dataset
import pandas as pd
import json

class NSynthDataSet_MelSpectrogram(Dataset):
    def __init__(self, meta_data_file, audio_dir, labels, instrument_label='instrument_family_str', sr=16000, hop_length=512):
        self.meta_data_file = meta_data_file
        self.audio_dir = audio_dir
        self.labels = labels
        self.instrument_label = instrument_label
        self.sr = sr
        self.hop_length = hop_length
        
        with open(os.path.join(meta_data_file)) as f:
            params = json.load(f)
            self.nsynth_meta_df = pd.DataFrame.from_dict(params)
            self.nsynth_meta_df = self.nsynth_meta_df.transpose()
        
        
    def __len__(self):
        return self.nsynth_meta_df.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx): #In case we get [0] instead of 0
            idx = idx.tolist()
        audio_file_name = self.nsynth_meta_df.iloc[idx].note_str + '.wav'
        # audio_category = self.labels.index(self.nsynth_meta_df.iloc[idx].instrument_family_str)
        audio_category = self.labels.index(self.nsynth_meta_df.iloc[idx][self.instrument_label])
        audio_data, _ = librosa.load(os.path.join(self.audio_dir, audio_file_name), sr=self.sr) 
        audio_stft_d = librosa.feature.melspectrogram(audio_data, sr=self.sr, hop_length=self.hop_length)
        audio_stft_d_dB = librosa.power_to_db(audio_stft_d, ref=np.max)
        audio_stft_d_dB = np.expand_dims(audio_stft_d_dB, axis=0)

        return audio_stft_d_dB, int(audio_category)

#audio_data is the amplitude array; audio_category is int ( 0 for Keyboard, 1 for Guitar, 2 for Bass)