import librosa
import os
import numpy as np
import torch as torch
from torch.utils.data import Dataset
import pandas as pd
import json

class NSynthDataSet_MFCC(Dataset):
    def __init__(self, meta_data_file, audio_dir, labels, instrument_label='instrument_family_str', sr=16000, n_mfcc=40):
        self.meta_data_file = meta_data_file
        self.audio_dir = audio_dir
        self.labels = labels
        self.instrument_label = instrument_label
        self.sr = sr
        self.mfcc = n_mfcc
        
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
        # audio_category = self.labels.index(self.nsynth_meta_df.iloc[idx].instrument_class_label)
        audio_category = self.labels.index(self.nsynth_meta_df.iloc[idx][self.instrument_label])
        audio_data, _ = librosa.load(os.path.join(self.audio_dir, audio_file_name), sr=self.sr) 

        mfccs = librosa.feature.mfcc(y=audio_data, sr=_, n_mfcc=self.mfcc)
        mfccs = np.expand_dims(mfccs, axis=0)
        return mfccs, int(audio_category)
#audio_data is the amplitude array; audio_category is int ( 0 for Keyboard, 1 for Guitar, 2 for Bass)