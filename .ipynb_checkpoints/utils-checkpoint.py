from transformers import WavLMForXVector
from optimizer import ScheduledOptim
import torch
from model import WavLMSV, TDNN, WavLMExtend
from os import listdir
import os
import numpy as np
def get_model(args, device, train=False):

    # load pretrained model 
    #model = WavLMSV(args).to(device)
    #model = TDNN(args).to(device)
    model = WavLMExtend(args).to(device)
    
    # to-do: add restore method
    if args.restore_step:
        ckpt_path = os.path.join(args.log_path, args.version, "{}.pth.tar".format(args.restore_step))
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    
    if train:
        scheduled_optim = ScheduledOptim(model, args.restore_step)
        model.train()
        if args.fix_wavLM:
            '''
            for param in model.wavlm.parameters():
                param.requires_grad = False
            '''
            model.wavlmxvector.freeze_base_model()
        return model, scheduled_optim
    
    model.eval()
    model.requires_grad_ = False
    return model


def to_device(data, device):
    if len(data) == 4:
        (wavs, attn_mask, speakers, langs) = data
        wavs = wavs.to(device)
        attn_mask = attn_mask.to(device)
        speakers = torch.from_numpy(speakers).long().to(device)
        langs = torch.from_numpy(langs).long().to(device)
        return [wavs, attn_mask, speakers, langs]
    
    elif len(data) == 2:
        (wavs, attn_mask) = data
        wavs = wavs.to(device)
        attn_mask = attn_mask.to(device)
        return [wavs, attn_mask]
    
    
def to_device_ADDA(data, device):
    if len(data) == 3:
        (wavs, attn_mask, src_embed) = data
        wavs = wavs.to(device)
        attn_mask = attn_mask.to(device)
        src_embed = torch.from_numpy(src_embed).to(device)
        return [wavs, attn_mask, src_embed]
    
    if len(data) == 2:
        (wavs, attn_mask) = data
        wavs = wavs.to(device)
        attn_mask = attn_mask.to(device)
        return [wavs, attn_mask]

def process_meta(dataset, args):
    dir_names = listdir(os.path.join(args.raw_path, dataset))
    speakers, basenames = [], []
    for name in dir_names:
        files =  listdir(os.path.join(args.raw_path, dataset, name))
        for file in files:
            if not file.endswith(".wav"):
                continue
            speakers.append(name)
            basenames.append(file[:-4])
    return speakers, basenames

def process_meta(args):
    speakers, basenames, datasets = [], [], []
    for dataset in args.datasets:
        dir_names = listdir(os.path.join(args.raw_path, dataset))
        for name in dir_names:
            files =  listdir(os.path.join(args.raw_path, dataset, name))
            for file in files:
                if not file.endswith(".wav"):
                    continue
                speakers.append(name)
                basenames.append(file[:-4])
                datasets.append(dataset)
    return speakers, basenames, datasets

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
    
def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask