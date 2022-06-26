import argparse
from os import listdir
import os
import json
import librosa
import numpy as np
from tqdm import tqdm
import random
import tgt
import yaml




def get_alignment(tier):
    sil_phones = ["sil", "sp", "spn"]
    
    start_time = 0
    end_time = 0
    flag = False
    
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text
        # Trim leading silences
        if flag == False:
            if p in sil_phones:
                continue
            else:
                start_time = s
                flag = True
        if p not in sil_phones:
            end_time = e
    return start_time, end_time
    
'''
    This code is used to process and create the test set
'''

def preprocess(args):
    
    # Get speakers to process
    with open(args.test_speakers_config, "r") as f:
        config = yaml.safe_load(f)
        test_spkers = config["speakers"]
    
    n_uttr_per_spker = dict([(spker, 0)for spker in test_spkers])
    testset_path = os.path.join(args.aishell3_dataset_path, "test", "wav")
    for spker in tqdm(listdir(testset_path)):
        if spker not in test_spkers:
            continue
            
        os.makedirs(os.path.join(args.wav_dir, spker), exist_ok=True)
        
        for file in listdir(os.path.join(testset_path, spker)):
            wav_path = os.path.join(os.path.join(testset_path, spker, file))
            wav, _ = librosa.load(wav_path, args.wav_sample_rate)
            wav = wav / max(abs(wav)) * 32768.0
            wav = wav.astype(np.float32)
            
            # Read TextGrid file to trim silence
            basename = file[:-4]
            tg_path = os.path.join(args.textgrid_path, spker, "{}.TextGrid".format(basename))
            if not os.path.exists(tg_path):
                continue
            textgrid = tgt.io.read_textgrid(tg_path)
            start, end = get_alignment(textgrid.get_tier_by_name("phones"))
            if start >= end:
                continue
            wav = wav[int(start*args.wav_sample_rate):int(end*args.wav_sample_rate)]
            
            # TODO: Skip short wavs?
            wav_len = len(wav)
            if wav_len < args.min_wav_length:
                continue
            
            # Save wav file
            n_uttr_per_spker[spker] += 1
            wav_filename = os.path.join(args.wav_dir, spker, "{}.npy".format(basename))
            np.save(wav_filename, wav)
       
    # Save Stat
    with open(os.path.join(args.preprocess_path, "stat.json"), "w") as f:
        f.write(json.dumps(
            {
                "total_uttrs": sum([n for n in n_uttr_per_spker.values()]),    
                "n_uttr_per_spker": n_uttr_per_spker,   
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
    parser.add_argument("--preprocess_path", type=str, default="./evaluation")
    parser.add_argument("--textgrid_path", type=str, default="/home/bai/data/local/VAE-TTS/preprocessed_data/AISHELL3/TextGrid")
    
    parser.add_argument("--wav_sample_rate", type=int, default=16000)
    parser.add_argument("--min_wav_length", type=int, default=48000) 
    parser.add_argument("--test_speakers_config", type=str, default="./evaluation/test_speakers.yaml")
    parser.add_argument("--aishell3_dataset_path", type=str, default="/home/bai/data/local/AISHELL-3")
    parser.add_argument("--wav_dir", type=str, default="./evaluation/wav")

    args = parser.parse_args()
    preprocess(args)