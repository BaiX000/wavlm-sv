from torch.utils.data import Dataset, DataLoader
import os
from os import listdir

import librosa
import numpy as np
from random import choice
from transformers import Wav2Vec2FeatureExtractor
import json
import random
from utils import pad_1D, pad_2D, get_mask_from_lengths

import torch
class Dataset(Dataset):
    def __init__(self, filename, args):
        
        #### settings ####
        self.args = args
        self.preprocess_path = args.preprocess_path
        self.sample_rate = 16000
        self.trim_wav = True
        self.max_wav_length = args.max_wav_length
        self.sort = True
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wavlm_pretrain_model)
        ##################
        
        self.speakers, self.basenames, self.datasets = self.process_meta(filename)
    
        with open(os.path.join(args.preprocess_path, "speakers.json"), 'r') as f:
            self.speaker_map = json.load(f)

        self.lang_map = {"LibriTTS":0, "AISHELL3":1}
        
    def __len__(self):
        return len(self.basenames)
    
    def __getitem__(self, idx):
        speaker = self.speakers[idx]
        speaker_id = self.speaker_map[speaker]
        basename = self.basenames[idx]
        dataset = self.datasets[idx]
        lang_id = self.lang_map[dataset]
        
        prefix = "data/source_data/train" if dataset == "LibriTTS" else "data/target_data/train"
        '''
        wavpath = os.path.join(prefix, "wav", "{}.npy".format(basename))
        wav = np.load(wavpath)
        '''
        mfccpath = os.path.join(prefix, "mfcc", "{}-mfcc.npy".format(basename[:-4]))
        # Read wav file 
        wav= np.load(mfccpath)
        wav = np.transpose(wav)
        
        # trim wav that is too long (random sample)
        if self.trim_wav:
            wav = self.sample_from_wav(wav, self.max_wav_length)
            
        sample = {
            "wav": wav,
            "speaker": speaker_id,
            "lang": lang_id,      
        }
        
        return sample
           
    def collate_fn(self, data):
        n = len(data)
        wavs = [data[i]["wav"] for i in range(n)]
        speakers = [data[i]["speaker"] for i in range(n)]
        langs = [data[i]["lang"] for i in range(n)]
        
        # wav
        '''
        output = self.feature_extractor(wavs, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        wavs = output.input_values
        attn_mask = output.attention_mask
        '''
        # mfcc
        
        attn_mask = [len(wav) for wav in wavs]
        attn_mask = torch.tensor(attn_mask)

        wavs = pad_2D(wavs)
        wavs = np.array(wavs)
        wavs = torch.from_numpy(wavs)
        
        speakers = np.array(speakers)
        langs = np.array(langs)
        
        return (wavs, attn_mask, speakers, langs)
    
    def process_meta(self, filename):
        with open(
            os.path.join(self.args.preprocess_path, filename), "r", encoding="utf-8"
        ) as f:
            speakers, basenames, datasets = [], [], []
 
            for line in f.readlines():
                s, b, d = line.strip("\n").split("|")
                speakers.append(s)
                basenames.append(b)
                datasets.append(d)
            return speakers, basenames, datasets
    
    
    def sample_from_wav(self, wav, max_length):
        if wav.shape[0] >= max_length:
            r = choice(range(0, len(wav) - max_length + 1))
            s = wav[r: r + max_length]
        else:
            s = wav
        return s
             
