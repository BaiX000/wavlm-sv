'''
extract():
    1. Extract each embeddings of uttrs in src domain dataset. i.e LibriTTS 
    2. Store the embeddings in "info-adda/src_embed" folder.
    3  Create "src_embed_filepath.txt" containing embeddings path.

preprocess():
    1. Creare a tgt_file containing target domain info. i.e AISHELL3
       info:(basename, path..)

'''

from tqdm import tqdm
import argparse
import torch
import tgt
import os 
from os import listdir
import json
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import random
from utils import to_device


def extract(args):
    
    device = torch.device('cuda')
    
    # Get Pretrained WavLM
    wavlmsv = WavLMForXVector.from_pretrained("./wavlm-base-plus-sv").to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./wavlm-base-plus-sv")

    # Read files from "info/wav"          
    src_out = []
    for file in tqdm(listdir(args.wav_dir)):
        if file[:3] == "SSB" or file[-4:] != ".npy":
            continue
            
        # Wav files are trimed in preprocess step, no need for doing it again
        wav_path = os.path.join(args.wav_dir, file)
        wav= np.load(wav_path)    

        # Get spker embed
        output = feature_extractor(wav, sampling_rate=args.wav_sample_rate, return_tensors="pt", padding=True)
        wav, attn_mask = output.input_values, output.attention_mask
        batch = (wav, attn_mask)
        batch = to_device(batch, device)
        with torch.no_grad():
            output = wavlmsv(*batch)
            xvector = output.embeddings.squeeze().cpu().numpy()

        # Save embedding
        basename = file[:-4]
        embed_filename = os.path.join(
            args.embed_dir,
            "{}-spker_embed.npy".format(basename),
        )
        np.save(embed_filename, xvector, allow_pickle=False)
        src_out.append(embed_filename)
            
    with open(os.path.join(args.preprocess_path, "src_embed_filepath.txt"), 'w', encoding="utf-8") as f:
        for l in src_out: 
            f.write(l + "\n")
    
def preprocess(args):
    tgt_out = []
    for file in tqdm(listdir(args.wav_dir)):
        if not file[:3] == "SSB" or file[-4:] != ".npy":
            continue
        basename = file[:-4]
        tgt_out.append(basename)
    
    random.shuffle(tgt_out)
    train = tgt_out[512:]
    val = tgt_out[:512]
    with open(os.path.join(args.preprocess_path, "train.txt"), "w", encoding="utf-8") as f:
        for l in train: 
            f.write(l + "\n")
            
    with open(os.path.join(args.preprocess_path, "val.txt"), "w", encoding="utf-8") as f:
        for l in val: 
            f.write(l + "\n")
    

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CPU training is not allowed."
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="./log")
    
    parser.add_argument("--wav_sample_rate", type=int, default=16000)

    # might change
    #parser.add_argument("--max_wav_length", type=int, default=64000) 
    parser.add_argument("--wav_dir", type=str, default="./info/wav")
    parser.add_argument("--embed_dir", type=str, default="./info-adda/src_embed")
    parser.add_argument("--preprocess_path", type=str, default="./info-adda")
    
    args = parser.parse_args()

    extract(args)
    preprocess(args)