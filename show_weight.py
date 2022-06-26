from tqdm import tqdm
import argparse
from utils import get_model, process_meta, get_alignment, to_device
import torch
import torch.nn as nn

import tgt
import os 
import json
from random import choice
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from sklearn import manifold
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt



class TSNE(nn.Module):
    def __init__(
        self, 
        args,
        n_component=2,
        perplexity=50, 
        n_iter=10000,
        init="random",
        random_state=0, 
        verbose=0,
    ):
        super(TSNE, self).__init__()
        self.args = args
        self.n_component = n_component
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        
    def forward(self, X):
        
        
        #t-SNE
        X_tsne = manifold.TSNE(
            n_components=self.n_component, 
            perplexity=self.perplexity,
            n_iter=self.n_iter,
            init=self.init,
            random_state=self.random_state,
            verbose=self.verbose
        ).fit_transform(X)

                    
        
        #Data Visualization
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
        fig = plt.figure(figsize=(10, 10))

        # 
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], "x", 
                     fontdict={'weight': 'bold', 'size': 5})
        plt.xticks([])
        plt.yticks([])
        
        file_name = "weight"
        file_name += ".pdf"

        fig.savefig(os.path.join(args.log_path, args.version, file_name))

def extract(args):
    

    # get SV model
    device = torch.device('cuda')
    wavlmsv = get_model(args, device, train=False)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wavlm_pretrain_model)
    weight = wavlmsv.spker_clsfir.weight
    weight = nn.functional.normalize(weight, dim=0)
    weight = weight.detach().cpu().numpy()

    tsne = TSNE(args)
    tsne(weight)
        

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CPU training is not allowed."
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=['LibriTTS', 'AISHELL3'],
        help="name of datasets",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    
    parser.add_argument("--raw_path", type=str, default="/home/bai/data/local/VAE-TTS/raw_data")
    parser.add_argument("--textgrid_path", type=str, default="/home/bai/data/local/VAE-TTS/preprocessed_data")
    parser.add_argument("--preprocess_path", type=str, default="./info")
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--wav_sample_rate", type=int, default=16000)


    # required
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--restore_step", type=int, required=True)

    # might change
    parser.add_argument("--max_wav_length", type=int, default=64000)
    parser.add_argument("--wavlm_pretrain_model", type=str, default="./wavlm-base-plus")
    
    # use this arg when using AMsoftmax
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--centralize", action='store_true')

    args = parser.parse_args()

    extract(args)