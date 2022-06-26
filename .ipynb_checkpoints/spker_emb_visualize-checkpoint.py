import argparse
import numpy as np
import torch.nn as nn
import os
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
        
        file_name = "spker_embed_{}".format(args.restore_step)
        if args.normalize:
            file_name += "_normalized"
        if args.centralize:
            file_name += "_centralized"
        if args.showCentralize:
            file_name += "_showCentralized"
        file_name += ".pdf"

        fig.savefig(os.path.join(args.log_path, args.version, file_name))
        
    
def main(args):
    name = "spker_embed_{}".format(args.restore_step)
    if args.normalize:
        name += "_normalized"
    if args.centralize:
        name += "_centralized"
    in_dir = os.path.join(args.log_path, args.version, name)
    
    
    # load gender info. of AISHELL3 & LibriTTS
    gender_map = {}
    with open(os.path.join(args.preprocess_path, "AISHELL3-spk-info.txt")) as f:
        lines = f.readlines()
        lines = lines[3:]
        for line in lines:
            spker, age , gender, region = line.strip().split('\t')
            gender_map[spker] = 0 if gender == "female" else 1  
            
    with open(os.path.join(args.preprocess_path, "LibriTTS-SPEAKERS.txt")) as f:
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
    for i, f in enumerate(os.listdir(in_dir)):
        if not f[-4:] == ".npy":
            continue
            

        spker = f.split("-")[0]
        if spker == "SSB1072":
            continue
        
        spker_lang = 1 if spker.startswith("SSB") else 0
        spker_embs.append(np.load(os.path.join(in_dir, f)))
        spker_names.append(spker)
        spker_genders.append(gender_map[spker])

        spker_langs.append(spker_lang)


    spker_embs = np.array(spker_embs)
    
    
    if args.showCentralize:
        print("Show Centralize")
        # get X by lang
        libritts_xs, aishell3_xs = [], []
        for (x, lang) in zip(spker_embs, spker_langs):
            libritts_xs.append(x) if lang==0 else aishell3_xs.append(x)
        libritts_xs = np.array(libritts_xs)
        aishell3_xs = np.array(aishell3_xs)
        '''
        #lang norm
        norm_spker_embs = []
        for (x, lang) in zip(spker_embs, spker_langs):
            if lang == 0:
                norm_spker_embs.append(
                    (x-libritts_xs.min(0))/(libritts_xs.max(0)-libritts_xs.min(0))
                )
            elif lang == 1:
                norm_spker_embs.append(
                    (x-aishell3_xs.min(0))/(aishell3_xs.max(0)-aishell3_xs.min(0))
                )
        spker_embeds = np.array(norm_spker_embs)
        
        
        # get normed X by lang 
        libritts_xs, aishell3_xs = [], []
        for (x, lang) in zip(spker_embs, spker_langs):
            libritts_xs.append(x) if lang==0 else aishell3_xs.append(x)
        libritts_xs = np.array(libritts_xs)
        aishell3_xs = np.array(aishell3_xs)
        '''
        # centralize           
        central_spker_embed = []
        a2l_bias = np.mean(libritts_xs, axis=0) - np.mean(aishell3_xs, axis=0)
        for (x, lang) in zip(spker_embs, spker_langs):
            if lang == 0:
                central_spker_embed.append(x)
            elif lang == 1:
                central_spker_embed.append(x+a2l_bias)
        spker_embs = np.array(central_spker_embed)
              
    tsne = TSNE(args)
    tsne(spker_embs, spker_names, spker_langs, spker_genders)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--preprocess_path", type=str, default="./info")
    parser.add_argument("--showCentralize", action='store_true')
    parser.add_argument("--restore_step", type=int)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--centralize", action='store_true')
    
    # centralize: take speaker embeds of ...._centralied
    # showCentralize: only show centralized spker embed (no change of data)
    
    args = parser.parse_args()
    main(args)