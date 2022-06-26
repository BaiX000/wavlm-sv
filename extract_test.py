from tqdm import tqdm
import argparse
from utils import get_model, process_meta, get_alignment, to_device
import torch
import tgt
import os 
import json
from random import choice
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


def sample_from_wav(wav, max_length):
    if wav.shape[0] >= max_length:
        r = choice(range(0, len(wav) - max_length + 1))
        s = wav[r: r + max_length]
    else:
        s = wav
    return s


def extract(args):
    

    # get SV model
    device = torch.device('cuda')
    #wavlmsv = get_model(args, device, train=False)
    wavlmsv = WavLMForXVector.from_pretrained("wavlm-base-plus-sv").to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wavlm_pretrain_model)

    # Read spkers 
    with open(os.path.join(args.preprocess_path, "speakers.json"), 'r') as f:
        speaker_map = json.load(f)
    spker_embed_dict = dict([(k, [])for k in speaker_map.keys()])
    
    speakers, basenames, datasets = process_meta(args)
    for i in tqdm(range(len(basenames))):
        (speaker, basename, dataset) = speakers[i], basenames[i], datasets[i]
        
        # some spkears are excluded from the subset
        if speaker not in speaker_map.keys():
            continue
        
        tg_path = os.path.join(args.textgrid_path, dataset, "TextGrid", speaker, "{}.TextGrid".format(basename))
        if not os.path.exists(tg_path):
            continue
            
        # get wav info 
        textgrid = tgt.io.read_textgrid(tg_path)
        start, end = get_alignment(textgrid.get_tier_by_name("phones"))
        if start >= end:
            continue  
            
        # Read and trim wav      
        wavpath = os.path.join(args.preprocess_path, "wav", "{}.npy".format(basename))
        # some wavs in raw_data may not in preprocess_path, because short wavs are skipped
        if not os.path.exists(wavpath):
            continue
        wav= np.load(wavpath)        
        # clip wav ?
        # wav = sample_from_wav(wav, args.max_wav_length)
        
        # Get spker embed
        output = feature_extractor(wav, sampling_rate=args.wav_sample_rate, return_tensors="pt", padding=True)
        wav, attn_mask = output.input_values, output.attention_mask
        batch = (wav, attn_mask)
        batch = to_device(batch, device)
        with torch.no_grad():
            output = wavlmsv(*batch)
            xvector = output.embeddings
            if args.normalize:
                xvector = torch.nn.functional.normalize(xvector, dim=-1)
        spker_embed_dict[speaker].append(xvector.squeeze().cpu().numpy())
        
    # save spker embeds
    dirname = "spker_embed_{}".format(str(args.restore_step))
    if args.normalize:
        dirname += "_normalized"
    if args.centralize:
        dirname += "_centralized"
        
    spker_embed_path = os.path.join(args.log_path, args.version, dirname)
    os.makedirs(spker_embed_path, exist_ok=True)
    
    
    # calculate mean speaker embedding
    spker_embed_mean_dict = dict()
    for spker in spker_embed_dict.keys():
        if len(spker_embed_dict[spker]) == 0:
            print("skip:{}".format(spker))
            continue
        print("Number of {} uttrs: {}".format(spker, len(spker_embed_dict[spker])))
        spker_embed_mean_dict[spker] = np.mean(spker_embed_dict[spker], axis=0)
    
    # if centralize, calculate a2l bias

    if args.centralize:
        libritts_xs, aishell3_xs = [], []
        for spker in spker_embed_mean_dict.keys():
            if spker[:3] == "SSB":
                aishell3_xs.append(spker_embed_mean_dict[spker])
            else:
                libritts_xs.append(spker_embed_mean_dict[spker])
        libritts_xs = np.array(libritts_xs)
        aishell3_xs = np.array(aishell3_xs)

        a2l_bias = np.mean(libritts_xs, axis=0) - np.mean(aishell3_xs, axis=0)
        
        # add bias to ailshell embed
        for spker in spker_embed_mean_dict.keys():
            if spker[:3] == "SSB":
                spker_embed_mean_dict[spker] = spker_embed_mean_dict[spker] + a2l_bias
    
    
    # save embeddings
    for spker in spker_embed_mean_dict.keys():
        spker_embed_filename = "{}-spker_embed.npy".format(spker)
        np.save(
                os.path.join(spker_embed_path, spker_embed_filename),
                spker_embed_mean_dict[spker],
                allow_pickle=False,
            )
        

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
    #parser.add_argument("--max_wav_length", type=int, default=64000) 
    parser.add_argument("--wavlm_pretrain_model", type=str, default="./wavlm-base-plus-sv")
    
    # use this arg when using AMsoftmax
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--centralize", action='store_true')


    
    args = parser.parse_args()

    extract(args)