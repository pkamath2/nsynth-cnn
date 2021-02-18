import librosa
from librosa.core import audio, magphase
import os
import numpy as np
import torch as torch
from torch.utils.data import Dataset
import pandas as pd
import json

class NSynthDataSet_IF(Dataset):
    def __init__(self, meta_data_file, audio_dir, labels, instrument_label='instrument_family_str', sr=16000):
        self.meta_data_file = meta_data_file
        self.audio_dir = audio_dir
        self.labels = labels
        self.instrument_label = instrument_label
        self.sr = sr
        
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

        inst_freq = self.encode(audio_data, _)
        return inst_freq, int(audio_category)
#audio_data is the amplitude array; audio_category is int ( 0 for Keyboard, 1 for Guitar, 2 for Bass)

    def stft(self, x):
        return librosa.core.stft(
                x,
                hop_length=512,
                win_length=1024,
                n_fft=1024)

    def mag_phase_angle(self, x):
        mag, ph = magphase(x)
        ph = np.angle(ph)
        out = np.stack([mag, ph])
        return out

    def safe_log(self, x):
        return torch.log(x + 1e-10)

    def safe_log_spec(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        if x.size(0) == 2:
            mag = x[0]
            ph = x[1]
            mlog = self.safe_log(mag)
            return torch.stack([mlog, ph], dim=0)
        elif x.size(0) == 1:
            return self.safe_log(x)

    def instantaneous_freq(self, specgrams):
        if specgrams.shape[0] != 2:
            mag = []
            ph = specgrams
        else:
            mag = specgrams[0]
            ph = specgrams[1]

        #first unwrap (phase keeps growing, no mod 2pi)
        uph = np.unwrap(ph, axis=1)
        #now take the difference between phase at this frame minus phase at previous frame
        uph_diff = np.diff(uph, axis=1)
        ifreq = np.concatenate([ph[:, :1], uph_diff], axis=1)

        if specgrams.shape[0] == 2:
            return np.stack([mag, ifreq/np.pi])
        else:
            return ifreq


    def encode(self, x,sr) :
        c = librosa.core.stft(
                x,
                hop_length=512,
                win_length=1024,
                n_fft=1024)
        mp= self.mag_phase_angle(c)
        #rm dc freq bin
        lspect=self.safe_log_spec(mp)
        return self.instantaneous_freq(lspect)