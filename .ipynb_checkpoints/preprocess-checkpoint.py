import argparse
from os import listdir
import os
import json
import librosa
import numpy as np
from tqdm import tqdm
import random
import tgt
from utils import process_meta, get_alignment


    
def preprocess(args):
    
    # create spker map 
    speakers = set()    
    for dataset in args.datasets:
        dir_names = listdir(os.path.join(args.raw_path, dataset))
        speakers.update(dir_names)      
    speakers = list(speakers)
    speakers.sort()
    speaker_map = dict()
    for i, spker in enumerate(speakers):
        speaker_map[spker] = i
    
    # store spker info 
    with open(os.path.join(args.preprocess_path, "speakers.json"), "w") as f:
        f.write(json.dumps(speaker_map))
        
    # sav wav data 
    os.makedirs(os.path.join(args.preprocess_path, "wav"), exist_ok=True)
    os.makedirs(os.path.join(args.preprocess_path, "mfcc"), exist_ok=True)
    
    
    speakers, basenames, datasets = process_meta(args)

    wav_lens = list()
    n_skip_short_wav = 0
    out = []
    n_wav_per_spker = dict((spker, 0) for spker in speakers)

    for i in tqdm(range(len(basenames))):
        (speaker, basename, dataset) = speakers[i], basenames[i], datasets[i]

        # Read wav file if textgrid exist (in order to trim silence wav)
        tg_path = os.path.join(args.textgrid_path, dataset, "TextGrid", speaker, "{}.TextGrid".format(basename))
        if not os.path.exists(tg_path):
            continue
        textgrid = tgt.io.read_textgrid(tg_path)
        start, end = get_alignment(textgrid.get_tier_by_name("phones"))
        if start >= end:
            continue        
            
        # Read and trim wav      
        wavpath = os.path.join(args.raw_path, dataset, speaker, "{}.wav".format(basename))
        wav, _ = librosa.load(wavpath, args.wav_sample_rate)
        wav = wav.astype(np.float32)
        wav = wav[int(args.wav_sample_rate * start) : int(args.wav_sample_rate * end)]
        
        # mfcc
        mfcc = librosa.feature.mfcc(wav, args.wav_sample_rate )
        
        # skip wav under ?? sec
        wav_length = len(wav)
        if wav_length < args.min_wav_length:
            n_skip_short_wav += 1
            continue
        # wav not skipped
        n_wav_per_spker[speaker] += 1
            
        
        # sav wav & mfcc &info
        wav_filename = os.path.join(args.preprocess_path, "wav", "{}-wav.npy".format(basename))
        np.save(wav_filename, wav)
        wav_lens.append(len(wav))
        
        mfcc_filename = os.path.join(args.preprocess_path, "mfcc", "{}-mfcc.npy".format(basename))
        np.save(mfcc_filename, mfcc)
        #out.append("|".join([speaker, basename, dataset]))
        
        
    # save files
    '''
    random.shuffle(out)
    train = out[args.val_size:]
    val = out[:args.val_size]
    
    with open(os.path.join(args.preprocess_path, "all.txt"), "w", encoding="utf-8") as f:
        for l in out: 
            f.write(l + "\n")
    
    with open(os.path.join(args.preprocess_path, "train.txt"), "w", encoding="utf-8") as f:
        for l in train: 
            f.write(l + "\n")
            
    with open(os.path.join(args.preprocess_path, "val.txt"), "w", encoding="utf-8") as f:
        for l in val: 
            f.write(l + "\n")
    '''
    
    with open(os.path.join(args.preprocess_path, "stat.json"), "w") as f:
        f.write(json.dumps(
            {
                "min_wav_len": min(wav_lens),
                "max_wav_len": max(wav_lens),
                "n_skip_short_wav": n_skip_short_wav,
                "total_wav": len(wav_lens),    
                "min_wav_per_spker": min(n_wav_per_spker.values()),
                "max_wav_per_spker": max(n_wav_per_spker.values()),
                "n_wav_per_spker": n_wav_per_spker,
                
            }
        ))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=['LibriTTS', 'AISHELL3'],
        help="name of datasets",
    )
    parser.add_argument("--preprocess_path", type=str, default="./info")
    parser.add_argument("--raw_path", type=str, default="/home/bai/data/local/VAE-TTS/raw_data")
    parser.add_argument("--textgrid_path", type=str, default="/home/bai/data/local/VAE-TTS/preprocessed_data")
    parser.add_argument("--val_size", type=int, default=512)
    
    parser.add_argument("--wav_sample_rate", type=int, default=16000)
    parser.add_argument("--min_wav_length", type=int, default=32000) 
    
    args = parser.parse_args()
    preprocess(args)
    
    # CUDA_VISIBLE_DEVICES=1 python3 preprocess.py --dataset AISHELL3 --preprocess_path ./data/target_data/train
    # CUDA_VISIBLE_DEVICES=1 python3 preprocess.py --dataset LibriTTS --preprocess_path ./data/src_data