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

import sys
sys.path.append("/home/bai/data/local/wavlm-sv")
from utils import get_model, to_device


def extract(args):
    
    device = torch.device('cuda')
    
    # Get SV model
    # Version 'pretrain' to test pretrain model from Hugging face
    if args.version == "WavLM-pretrained":
        wavlmsv = WavLMForXVector.from_pretrained("./wavlm-base-plus-sv").to(device)
    else:
        wavlmsv = get_model(args, device, train=False)
        
        
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wavlm_pretrain_model)

    # Read files form "evaluation/wav"             
    out_dir = os.path.join(args.embed_dir, "{}_{}_{}_{}".format(args.version, args.restore_step, args.domain, args.input_type))
    os.makedirs(out_dir, exist_ok=True)
    for file in tqdm(listdir(args.wav_dir)):    

        # Get spker embed
        if args.input_type == "wav":
            wav_path = os.path.join(args.wav_dir, file)
            wav= np.load(wav_path) 
            output = feature_extractor(wav, sampling_rate=args.wav_sample_rate, return_tensors="pt", padding=True)
            wav, attn_mask = output.input_values, output.attention_mask
            batch = (wav, attn_mask)
            
        elif args.input_type == "mfcc":
            mfcc_path = os.path.join(args.wav_dir, file)
            mfcc = np.load(mfcc_path)
            mfcc = mfcc.transpose()
            mfcc = torch.from_numpy(np.array([mfcc]))
            src_len = torch.tensor([len(mfcc)])
            batch = (mfcc, src_len)
        batch = to_device(batch, device)
        
        with torch.no_grad():
            output = wavlmsv(*batch)
            xvector = output.embeddings.squeeze().cpu().numpy()
            #xvector = output[2].squeeze().cpu().numpy()
        # Save embedding
        basename = file[:-4]
        embed_filename = os.path.join(
            out_dir,
            "{}-spker_embed.npy".format(basename),
        )
        np.save(embed_filename, xvector, allow_pickle=False)
                

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CPU training is not allowed."
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="./log")   
    parser.add_argument("--wav_sample_rate", type=int, default=16000)
    
    # required
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--restore_step", type=int, required=True)

    parser.add_argument("--wavlm_pretrain_model", type=str, default="./wavlm-base-plus")  
    
    parser.add_argument("--embed_dir", type=str, default="./evaluation/embed")
    
    # Set before RUN !
    parser.add_argument("--domain", type=str, required=True) # [source, target]
    parser.add_argument("--input_type", type=str, required=True) # [wav, mfcc]
    
    # Enter the model's domain
    parser.add_argument("--preprocess_path", type=str, default="data/mix_data/train") # [wav, mfcc]

    args = parser.parse_args()

    args.wav_dir = "./data/{}_data/train/{}".format(args.domain, args.input_type)
    
    extract(args)