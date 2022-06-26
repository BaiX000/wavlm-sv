import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import to_device

from dataset import Dataset
from loss import WavLMSVLoss, WavLMExtendLoss


def evaluate(device, model, step, args, len_losses=3):
    dataset = Dataset("val.txt", args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    
    Loss = WavLMSVLoss(args).to(device)
    #Loss = WavLMExtendLoss(args).to(device)
    
    
    loss_sums = [0 for _ in range(len_losses)]
    #loss_sums = 0
    spker_accuracy_sum = 0.0
    lang_accuracy_sum = 0.0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.no_grad():
            output = model(*batch)
            # !m
            losses, lang_accuracy, spker_accuracy = Loss(batch[-2:], output)
            total_loss = losses[0]
            '''
            total_loss, spker_accuracy = Loss(output, batch[-2])
            '''
        for i in range(len(losses)):
            loss_sums[i] += losses[i].item() * len(batch[0])
        spker_accuracy_sum += spker_accuracy
        lang_accuracy_sum += lang_accuracy
        '''
        loss_sums += total_loss.item() * len(batch[0])
        spker_accuracy_sum += spker_accuracy
        '''
        
    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]
    speaker_accuracy_mean = spker_accuracy_sum / len(loader)
    lang_accuracy_mean = lang_accuracy_sum /len(loader)
    
    message = "Validation Step {}, Total Loss: {:.4f}, Spker Loss: {:.4f}, Lang Loss {:.4f}, Spker Accuracy:{:.4f}, Lang Accuracy:{:.4f}".format(
        *([step] + [l for l in loss_means] + [speaker_accuracy_mean, lang_accuracy_mean])
    )
    '''
    message = "Validation Step {}, Total Loss: {:.4f}, Spker Accuracy:{:.4f}".format(step, loss_sums/len(dataset), spker_accuracy_sum / len(loader))
    '''
    return message
    

# ADDA Evaluation
'''
from torch.utils.data import Dataset, DataLoader


class DatasetEval(Dataset, wav_dir="./evaluation/wav"):
    def __init__(self):
        basenames = []
        spkers = []
        for spker in listdir(wav_dir):
            files = istdir(os.path.join(wav_dir, spker))
            if len(files) == 0:
                continue
            for file in files:
                basename = file[:-4]
                basename.append(basename)
                spkers.append(spker)
                
        self.wav_dir = wav_dir
        self.basenames = basenames
        self.spkers = spkers
        
    def __len__(self):
        return len(self.basenames)
        
    def __getitem__(self, idx):
        basename = self.basenames[idx]
        spker = self.spkers[idx]
        wav_path = os.path.join(self.wav_dir, spker, "{}.npy".format(basename))
        wav = np.load(wav_path)
        # trim wavs?
        
        sample = {
            "wav": wav,
            "spker": spker,
        }
    
    
    def collate_fn(self, data):
        n = len(data)
        pass

'''
'''
from os import listdir
from transformers import Wav2Vec2FeatureExtractor
from utils import to_device_ADDA
from os import listdir
import numpy as np
from tqdm import tqdm
from numpy import dot

def evaluate_ADDA(device, model, step, args):
    
    # settings
    wav_dir = "./evaluation/wav"
    n_uttr_skip_spker = 5
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("wavlm-base-plus-sv")
    
    # Step 1: extract embedding throught current model (Generator)
    embeddings = []
    for spker in listdir(wav_dir):
        files = listdir(os.path.join(wav_dir, spker))
        if len(files) < n_uttr_skip_spker:
            continue
        for file in files:
            wav_path = os.path.join(wav_dir, spker, file)
            wav = np.load(wav_path)
            
            output = feature_extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
            wav, attn_mask = output.input_values, output.attention_mask
            batch = (wav, attn_mask)
            batch = to_device_ADDA(batch, device)
            with torch.no_grad():
                embed = model(*batch).embeddings
                embed = embed.squeeze().cpu().numpy()
                embed = embed / np.linalg.norm(embed)
                embeddings.append((spker, embed))
   
    # Step 2: compute EER
    
    n = len(embeddings)
    sim = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        row_spker, row_embed = embeddings[i]
        for j in range(n):
            col_spker, col_embed = embeddings[j]
            sim[i][j] = dot(row_embed, col_embed)
    
    min_DIFF = 1
    for TRESHOLD in np.arange(0.86, 1.0, 0.005):
        FA, FR, IN, OUT = 0, 0, 0, 0
        for i in range(n):
            row_spker, _ = embeddings[i]
            for j in range(n):
                if i == j:
                    continue
                col_spker, _ = embeddings[j]
                similarity = sim[i][j]
                if row_spker == col_spker:
                    IN += 1
                    if similarity < TRESHOLD:
                        FR += 1
                else:
                    OUT += 1
                    if similarity >= TRESHOLD:
                        FA += 1
        FRR = FR/IN
        FAR = FA/OUT
        print(TRESHOLD, FRR, FAR)
        
        diff = abs(FRR-FAR)
        if diff < min_DIFF:
            min_DIFF = diff
            INFO = [TRESHOLD, FRR, FAR] 
        if FRR > FAR:
            break
            
    return "Evaluation Step {}, ERR: {:.4f} ~ {:.4f}, Treshold: {:.3f}".format(step, INFO[1], INFO[2], INFO[0])
'''