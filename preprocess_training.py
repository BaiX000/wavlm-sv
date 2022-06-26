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
    
    # Read files
    in_dir = os.listdir(os.path.join(args.preprocess_path)):
    for files in in_dir:
        pass
        
        
    # Filter out the spkers to skip
    spker_to_skip = []
    for spker in stats["n_wav_per_spker"].keys():
        if stats["n_wav_per_spker"][spker] < args.n_uttr_under_to_skip:
            spker_to_skip.append(spker)
    
    
    # Read all.txt file to creat new subset
    out = []
    with open(os.path.join(args.preprocess_path, "all.txt"), 'r') as f:
        lines = f.readlines()
        print("\"all.txt\" contains {} lines of data".format(len(lines)))
        for line in lines:
            line = line.rstrip()
            if line.split("|")[0] in spker_to_skip:
                continue
            out.append(line)
    print("After processing, {} speakers {} lines of data are skipped"\
          .format(len(spker_to_skip), len(lines)-len(out)))
    
    # creat new speaker map
    spker_map = {}
    i = 0
    for spker in stats["n_wav_per_spker"].keys():
        if spker in spker_to_skip:
            continue
        spker_map[spker] = i
        i += 1
    
    # store data
    train = out[args.val_size:]
    val = out[:args.val_size]
    
    with open(os.path.join(args.preprocess_path, "train.txt"), "w", encoding="utf-8") as f:
        for l in train: 
            f.write(l + "\n")
            
    with open(os.path.join(args.preprocess_path, "val.txt"), "w", encoding="utf-8") as f:
        for l in val: 
            f.write(l + "\n")
    with open(os.path.join(args.preprocess_path, "subset_stat.json"), "w") as f:
        f.write(json.dumps(
            {
                "n_uttr_under_to_skip": args.n_uttr_under_to_skip,
                "spker_to_skip":spker_to_skip,
            }
        ))
           
    with open(os.path.join(args.preprocess_path, "speakers.json"), "w") as f:
        f.write(json.dumps(spker_map))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--preprocess_path", type=str, default="./data/src_data")
    parser.add_argument("--val_size", type=int, default=512)
    parser.add_argument("--n_uttr_under_to_skip", type=int, default=20) 
    
    args = parser.parse_args()
    preprocess(args)