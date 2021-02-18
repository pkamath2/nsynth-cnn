from dataset.NSynthDataSet_MelSpectrogram import NSynthDataSet_MelSpectrogram
import torch 

from dataset.NSynthDataSet_MFCC import NSynthDataSet_MFCC
from dataset.NSynthDataSet_MelSpectrogram import NSynthDataSet_MelSpectrogram
from dataset.NSynthDataSet_IF import NSynthDataSet_IF
from dataset.NSynthDataSet_MelSpectrogram_harmonics import NSynthDataSet_MelSpectrogram_harmonics


def load_dataset(type, meta_data_file, audio_dir, batch_size, labels, sample_rate, n_mfcc=40, hop_length=512):
    loader = ''
    if type == 'mfcc':
        ds = NSynthDataSet_MFCC(meta_data_file=meta_data_file, audio_dir=audio_dir, labels=labels, sr=sample_rate, n_mfcc=n_mfcc)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    if type == 'mel':
        ds = NSynthDataSet_MelSpectrogram(meta_data_file=meta_data_file, audio_dir=audio_dir, labels=labels, sr=sample_rate, hop_length=hop_length)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    if type == 'if':
        ds = NSynthDataSet_IF(meta_data_file=meta_data_file, audio_dir=audio_dir, labels=labels, sr=sample_rate)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    if type == 'mel_harmonics':
        ds = NSynthDataSet_MelSpectrogram_harmonics(meta_data_file=meta_data_file, audio_dir=audio_dir, labels=labels, sr=sample_rate, hop_length=hop_length)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader
