import argparse
import numpy as np
import torch.nn as nn
import os
from sklearn import manifold
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import librosa
from tqdm import tqdm
import torch
import sys
sys.path.append("/home/bai/data/local/wavlm-sv")
from utils import get_model, to_device

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
        
    def forward(self, X, spker_names, spker_langs, spker_genders):
        
        
        #t-SNE
        X_tsne = manifold.TSNE(
            n_components=self.n_component, 
            perplexity=self.perplexity,
            n_iter=self.n_iter,
            init=self.init,
            random_state=self.random_state,
            verbose=self.verbose
        ).fit_transform(X)

        #Color map
        lang_map = ["green", "blue"]
        gender_map = ["hotpink", "#88c999"]
        lang_text = ["O", "X"]
        gender_text = ["g", "b"]
                    
        
        #Data Visualization
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
        fig = plt.figure(figsize=(10, 10))

        # 
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], gender_text[spker_genders[i]], color=lang_map[spker_langs[i]], 
                     fontdict={'weight': 'bold', 'size': 5})
        plt.xticks([])
        plt.yticks([])
        
        file_name = "spker_embed.pdf"

        fig.savefig(os.path.join(args.out_visualize_path, file_name))
        
    
def visualize(args):
    model_basename = args.embed_dir.split("/")[-1]
    out_spker_embed_path = os.path.join(args.spker_embed_dir, model_basename)
    out_visualize_path = os.path.join(args.visualize_dir, model_basename)
    os.makedirs(out_visualize_path, exist_ok=True)
    args.out_visualize_path = out_visualize_path
    
    # Load Gender info.: AISHELL3 & LibriTTS
    gender_map = {}
    with open(os.path.join("./info", "AISHELL3-spk-info.txt")) as f:
        lines = f.readlines()
        lines = lines[3:]
        for line in lines:
            spker, age , gender, region = line.strip().split('\t')
            gender_map[spker] = 0 if gender == "female" else 1  
            
    with open(os.path.join("./info", "LibriTTS-SPEAKERS.txt")) as f:
        lines = f.readlines()
        lines = lines[12:]
        for line in lines:
            spker, gender, subset = line.strip().split('|')[:3]
            spker, gender, subset = spker.strip(), gender.strip(), subset.strip()
            if subset != "train-clean-360":
                continue
            gender_map[spker] = 0 if gender == "F" else 1
    
    
    spker_embs = []
    spker_names = []
    spker_langs = []
    spker_genders = []
       
    for i, f in enumerate(os.listdir(out_spker_embed_path)):
        if not f[-4:] == ".npy":
            continue
        basename = f.split("-")[0]
        spker = basename[:7] if basename[:3] == "SSB" else basename.split("_")[0] 

        spker_lang = 1 if spker.startswith("SSB") else 0
        spker_embs.append(np.load(os.path.join(out_spker_embed_path, f)))
        spker_names.append(spker)
        spker_genders.append(gender_map[spker])
        spker_langs.append(spker_lang)

    spker_embs = np.array(spker_embs)
              
    tsne = TSNE(args)
    tsne(spker_embs, spker_names, spker_langs, spker_genders)
    
    
def extract(args):
    
    # settings
    model_basename = args.embed_dir.split("/")[-1]
    out_spker_embed_path = os.path.join(args.spker_embed_dir, model_basename)
    
    os.makedirs(out_spker_embed_path, exist_ok=True)
    
    
    
    # Get Model 
    device = torch.device('cuda')
    wavlmsv = get_model(args, device, train=False)
    #wavlmsv = WavLMForXVector.from_pretrained(args.wavlm_pretrain_model).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wavlm_pretrain_model)
    
    # Get files: LibriTTS + AISHELL3
    wav_paths = []
    wav_paths += [os.path.join("data/source_data/train/wav", f) for f in os.listdir("data/source_data/train/wav")]
    wav_paths += [os.path.join("data/target_data/train/wav", f) for f in os.listdir("data/target_data/train/wav")]
    wav_paths += [os.path.join("data/source_data/test/wav", f) for f in os.listdir("data/source_data/test/wav")]
    wav_paths += [os.path.join("data/target_data/test/wav", f) for f in os.listdir("data/target_data/test/wav")]

    # Create spkear to embeds dictionary
    spker_to_embeds = dict()
    for wav_path in tqdm(wav_paths):
        f = wav_path.split("/")[-1]
        if not f[-4:] == ".npy":
            continue
        basename = f.split("-")[0]
        spker = basename[:7] if basename[:3] == "SSB" else basename.split("_")[0]
        
        # load np file
        wav = np.load(wav_path)
        output = feature_extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        wav, attn_mask = output.input_values, output.attention_mask
        batch = (wav, attn_mask)
        batch = to_device(batch, device)
        
        with torch.no_grad():
            output = wavlmsv(*batch)
            embed = output.embeddings.squeeze().cpu().numpy()
                    
        if spker not in spker_to_embeds.keys():
            spker_to_embeds[spker] = []           
        spker_to_embeds[spker].append(embed)
        
    print("Total Speakers: {}".format(len(spker_to_embeds.keys())))

    # save embeddings
    for spker in spker_to_embeds.keys():
        spker_embed_filename = "{}-spker_embed.npy".format(spker)
        np.save(
                os.path.join(out_spker_embed_path, spker_embed_filename),
                np.mean(spker_to_embeds[spker], axis=0),
                allow_pickle=False,
            )
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="./log")   
    parser.add_argument("--wavlm_pretrain_model", type=str, default="./wavlm-base-plus-sv")  
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--restore_step", type=int)
    
    parser.add_argument("--visualize_dir", type=str, default="./evaluation/visualize")
    parser.add_argument("--spker_embed_dir", type=str, default="./evaluation/spker_embed")

    # Enter the model's domain
    parser.add_argument("--preprocess_path", type=str, default="data/mix_data/train") # [wav, mfcc]
    
    # change every time
    parser.add_argument("--embed_dir", type=str, default="./evaluation/embed/pretrained")

    
    args = parser.parse_args()
    extract(args)
    #visualize(args)