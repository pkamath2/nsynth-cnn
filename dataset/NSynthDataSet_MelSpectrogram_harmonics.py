import librosa
import os
import numpy as np
import torch as torch
from torch.utils.data import Dataset
import pandas as pd
import json
import time

class NSynthDataSet_MelSpectrogram_harmonics(Dataset):
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
        start1 = time.perf_counter()
        if torch.is_tensor(idx): #In case we get [0] instead of 0
            idx = idx.tolist()
        audio_file_name = self.nsynth_meta_df.iloc[idx].note_str + '.wav'
        # audio_category = self.labels.index(self.nsynth_meta_df.iloc[idx].instrument_class_label)
        audio_category = self.labels.index(self.nsynth_meta_df.iloc[idx][self.instrument_label])
        audio_data, _ = librosa.load(os.path.join(self.audio_dir, audio_file_name), sr=self.sr) 
        
        audio_d_hpss = librosa.effects.hpss(audio_data)
        audio_data_harmonic = audio_d_hpss[0]
        audio_data_percussive = audio_d_hpss[1]
        audio_stft_harmonic = librosa.feature.melspectrogram(audio_data_harmonic, sr=self.sr, hop_length=self.hop_length)
        audio_stft_percussive = librosa.feature.melspectrogram(audio_data_percussive, sr=self.sr, hop_length=self.hop_length)
        audio_stft_harmonic_dB = librosa.power_to_db(audio_stft_harmonic, ref=np.max)
        audio_stft_percussive_dB = librosa.power_to_db(audio_stft_percussive, ref=np.max)

        data = np.stack([audio_stft_harmonic_dB, audio_stft_percussive_dB])
        return data, int(audio_category)

#audio_data is the amplitude array; audio_category is int ( 0 for Keyboard, 1 for Guitar, 2 for Bass)