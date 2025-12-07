import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio
import warnings
import logging
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
import whisper
import tarfile
import pickle
import tempfile
import librosa

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead")


def process_Rawboost_feature(feature, sr, args, algo):
    # Data process by Convolutive noise (1st algo)
    if algo == 1:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)

    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:

        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW,
                                     args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
                                     args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG,
                                     args.maxG, sr)

        # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

        # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW,
                                     args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

        # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW,
                                     args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

        # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:

        feature1 = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                         args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                         args.maxBiasLinNonLin, sr)
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)  # normalized resultant waveform

    # original data without Rawboost processing
    else:

        feature = feature

    return feature




def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):

    if x.ndim > 1:
        x = x.mean(axis=0)

    x = np.squeeze(x)
    x_len = x.shape[0]
    if x_len == 0:
        return np.zeros(max_len, dtype=np.float32)

    if x_len >= max_len:
        stt = np.random.randint(max(x_len - max_len + 1, 1))
        return x[stt:stt + max_len]

    repeat = max_len // x_len
    remainder = max_len % x_len
    return np.concatenate([x] * repeat + [x[:remainder]])





class Dataset_(Dataset):
    def __init__(self, dir_meta, root_dir, cut=64600, use_cache=False,args=None,whisper=False):
        self.full_path_list = []
        self.label_list = []
        self.label_map = {'bona-fide':1
            ,'bonafide': 1, 'spoof': 0
            , 'genuine': 1, 'fake': 0
            , 'real': 1, 'spoof':0}
        self.cut = cut
        self.use_cache = use_cache
        self.algo = args.algo
        self.args = args
        self.whisper = whisper
        with open(dir_meta, 'r') as f:
            for line in f:
                rel_path, label_str = line.strip().split()
                full_path = os.path.join(root_dir, rel_path)
                self.full_path_list.append(full_path)
                self.label_list.append(self.label_map[label_str])


        self.cached_data = {} if self.use_cache else None

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index, target_sr=16000):
        filepath = self.full_path_list[index]
        y = self.label_list[index]

        if self.use_cache and filepath in self.cached_data:
            x_inp = self.cached_data[filepath]
        else:

            if filepath.lower().endswith('.mp3'):
                waveform, original_sr = librosa.load(filepath, sr=target_sr)
                waveform = waveform.squeeze()
            else:
                waveform, original_sr = torchaudio.load(filepath)
                if original_sr != target_sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
                    waveform = resampler(waveform)
                waveform = waveform.squeeze().numpy()
            waveform = process_Rawboost_feature(waveform,target_sr,self.args,self.algo)
            waveform = pad_random(waveform, self.cut)
            x_inp = Tensor(waveform)
            if self.use_cache:
                self.cached_data[filepath] = x_inp
        return x_inp, y, filepath


class Dataset_0(Dataset):
    def __init__(self, dir_meta, root_dir, cut=64600, use_cache=False,args=None,whisper=False):
        self.full_path_list = []
        self.label_list = []
        self.label_map = {'bona-fide':1,'bonafide': 1
            , 'spoof': 0, 'genuine': 1
            , 'fake': 0, 'real': 1
            , 'spoofed':0}
        self.cut = cut
        self.args = args
        self.whisper = whisper
        with open(dir_meta, 'r') as f:
            for line in f:
                rel_path, label_str = line.strip().split()
                full_path = os.path.join(root_dir, rel_path)
                self.full_path_list.append(full_path)
                self.label_list.append(self.label_map[label_str])


    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index, target_sr=16000):
        filepath = self.full_path_list[index]
        y = self.label_list[index]
        if filepath.lower().endswith('.mp3') or filepath.lower().endswith('.flac'):
            waveform, original_sr = librosa.load(filepath, sr=target_sr)
            waveform = waveform.squeeze()
        else:
            waveform, original_sr = torchaudio.load(filepath)

            if original_sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
                waveform = resampler(waveform)
            waveform = waveform.squeeze().numpy()
        if self.whisper:
            waveform = whisper.pad_or_trim(waveform)
            mel = whisper.log_mel_spectrogram(waveform,n_mels=128)
            return mel, y, filepath

        waveform = pad_random(waveform, self.cut)
        return waveform, y, filepath

class Dataset_composition(Dataset):
    def __init__(self, dir_meta, root_dir, cut=64600, use_cache=False,args=None,whisper=False):
        self.full_path_list = []
        self.label_list = []
        self.cut = cut
        self.args = args
        self.whisper = whisper
        self.label_map = {"spoof_spoof": 4, "bonafide_spoof": 3
            , "spoof_bonafide": 2, "bonafide_bonafide": 1, "bonafide_0": 0}
        with open(dir_meta, 'r') as f:
            for line in f:
                rel_path,_,_, label_str = line.strip().split()
                full_path = os.path.join(root_dir, rel_path)
                self.full_path_list.append(full_path)
                self.label_list.append(int(self.label_map[label_str]))


    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index, target_sr=16000):
        filepath = self.full_path_list[index]
        y = self.label_list[index]
        if filepath.lower().endswith('.mp3') or filepath.lower().endswith('.flac'):
            waveform, original_sr = librosa.load(filepath, sr=target_sr)
            waveform = waveform.squeeze()
        else:
            waveform, original_sr = torchaudio.load(filepath)

            if original_sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
                waveform = resampler(waveform)
            waveform = waveform.squeeze().numpy()
        if self.whisper:
            waveform = whisper.pad_or_trim(waveform)
            mel = whisper.log_mel_spectrogram(waveform,n_mels=128)
            return mel, y, filepath

        waveform = pad_random(waveform, self.cut)
        return waveform, y, filepath

class Dataset_api(Dataset):
    def __init__(self, dir_meta, root_dir, cut=64600, use_cache=False,args=None,whisper=False):
        self.full_path_list = []
        self.label_list = []

        self.cut = cut
        self.args = args
        self.whisper = whisper
        with open(dir_meta, 'r') as f:
            for line in f:
                rel_path, label_str = line.strip().split()
                full_path = os.path.join(root_dir, rel_path)
                self.full_path_list.append(full_path)
                self.label_list.append(int(label_str.strip()))


    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index, target_sr=16000):
        filepath = self.full_path_list[index]
        y = self.label_list[index]
        if filepath.lower().endswith('.mp3') or filepath.lower().endswith('.flac'):
            waveform, original_sr = librosa.load(filepath, sr=target_sr)
            waveform = waveform.squeeze()
        else:
            waveform, original_sr = torchaudio.load(filepath)

            if original_sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
                waveform = resampler(waveform)
            waveform = waveform.squeeze().numpy()
        if self.whisper:
            waveform = whisper.pad_or_trim(waveform)
            mel = whisper.log_mel_spectrogram(waveform,n_mels=128)
            return mel, y, filepath

        waveform = pad_random(waveform, self.cut)
        return waveform, y, filepath