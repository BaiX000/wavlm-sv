import argparse
import os
from os import listdir

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import random
import json

def main(args):
    embed_path = os.path.join("./evaluation/embed", args.basepath)
    embeds = []
    basenames = []
    for f in listdir(embed_path):
        f_path = os.path.join(embed_path, f)
        basename = f.split('-')[0]
        embed = np.load(f_path)
        embeds.append(embed)
        basenames.append(basename)
        
    X = np.array(embeds)
    print(X.shape)
    
    #cluster = AgglomerativeClustering(n_clusters=args.n_cluster, affinity='cosine', linkage='average')
    cluster = AgglomerativeClustering(n_clusters=args.n_cluster, linkage='ward')
    cluster.fit(X)
    labels = cluster.labels_
    
    out = [ "|".join([str(s), b+"-wav", "AISHELL3"]) for (s, b) in zip(labels, basenames)]
    random.shuffle(out)
    train = out[512:]
    val = out[:512]
    with open(os.path.join(args.out_dir, "pseudo-{}-train.txt".format(args.n_cluster)), 'w') as f:
        for l in train:
            f.write(l+'\n')
            
    with open(os.path.join(args.out_dir, "pseudo-{}-val.txt".format(args.n_cluster)), 'w') as f:
        for l in val:
            f.write(l+'\n')
    
    with open(os.path.join(args.out_dir, "pseudo-{}-speakers.json".format(args.n_cluster)), 'w') as f:
        f.write(json.dumps(dict([str(k), int(k)] for k in set(labels))))
        
    n_per_class = dict([str(k), 0] for k in set(labels))   
    for c in labels:
        n_per_class[str(c)] += 1
    
    with open(os.path.join(args.out_dir, "pseudo-{}-stat.json".format(args.n_cluster)), 'w') as f:
        f.write(json.dumps(n_per_class))
        
    '''
    plt.figure(figsize=(100, 70))
    dend = dendrogram(linkage(X, method='average', metric='cosine'))
    plt.savefig("dend.png")
    '''
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cluster", type=int, default=100)
    parser.add_argument("--basepath", type=str, default="WavLM-pretrained_0_target_wav_train")
    parser.add_argument("--out_dir", type=str, default="data/target_data/train")
    args = parser.parse_args()
    
    main(args)