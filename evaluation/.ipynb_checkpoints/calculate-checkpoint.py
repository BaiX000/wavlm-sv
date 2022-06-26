from tqdm import tqdm
import argparse
import torch
import tgt
import os 
from os import listdir
import json
import librosa
import numpy as np
from numpy import dot
from numpy.linalg import norm

def main(args):
    
    
    # Read Embeddings            
    in_dir = os.path.join(args.embed_dir, "{}_{}_{}_{}".format(args.version, args.restore_step, args.domain, args.input_type))
    
    embeddings = []
    for file in listdir(in_dir):
        spker = file[:7] if file[:3] == "SSB" else file.split('_')[0]
        embed_path = os.path.join(in_dir, file)
        xvector = np.load(embed_path)    
        embeddings.append((spker, xvector))
            
    # create N*N array to store cosine smilarity
    n = len(embeddings)
    sim = [[0 for _ in range(n)] for _ in range(n)]
    for i in tqdm(range(n)):
        row_spker, row_embed = embeddings[i]
        for j in range(i, n):
            col_spker, col_embed = embeddings[j]
            sim[i][j] = dot(row_embed, col_embed)/(norm(row_embed)*norm(row_embed))
    
    
    min_DIFF = 1
    # DFC = C_fa * FAR * (1-p_target) + C_fr * FRR * p_target
    C_fa, C_fr = 1, 10
    p_target = 0.01
    min_DCF = 100
    for TRESHOLD in np.arange(0.095, 1.0, 0.0005):
        FA, FR, IN, OUT = 0, 0, 0, 0
        for i in range(n):
            row_spker, _ = embeddings[i]
            for j in range(i, n):
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
        #min_DCF = C_fa * FAR * (1-p_target) + C_fr * FRR * p_target
        
        print("Treshold {:.4f}, FRR {:.5f}, FAR {:.5f}, min_DCF {:.5f}".format(TRESHOLD, FRR, FAR, min_DCF))
        
        DIFF = abs(FRR-FAR)
        if DIFF < min_DIFF:
            INFO = (FRR, FAR, TRESHOLD) 
            min_DIFF = DIFF
        #if FRR > FAR:
         #   break
        
    '''
    message = "EER: {:.5f}~{:.5f}, THRESHOLD:{:.5f}".format(INFO[0], INFO[1], INFO[2])
    print(message)
    with open(os.path.join(args.log_dir, "{}_{}_{}_{}-EER.txt").format(args.version, args.restore_step, args.domain, args.input_type), 'a') as f:
        f.write(message + '\n')
    '''
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--embed_dir", type=str, default="./evaluation/embed")
    parser.add_argument("--n_uttr_skip_spker", type=int, default=5)
    parser.add_argument("--log_dir", type=str, default="./evaluation/log")
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--input_type", type=str, required=True)
    
    args = parser.parse_args()
    print("Skip speaker who's uttr number is under: {} uttrs".format(args.n_uttr_skip_spker))
    main(args)