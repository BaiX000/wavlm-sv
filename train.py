from transformers import Wav2Vec2FeatureExtractor
import torch
import os 
import librosa
import numpy as np
from tqdm import tqdm

import argparse
from utils import get_model, to_device
from dataset import Dataset
from torch.utils.data import DataLoader

from torch.cuda import amp
from loss import WavLMSVLoss, WavLMExtendLoss
    
from torch.utils.tensorboard import SummaryWriter
from evaluate import evaluate

def train(args):
    device = torch.device('cuda')
    
    # Get dataset
    dataset = Dataset("train.txt", args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    # Prepare model
    model, optimizer = get_model(args, device, train=True)
    scaler = amp.GradScaler()
    Loss = WavLMSVLoss(args).to(device)
    #Loss = WavLMExtendLoss(args).to(device)
    # --settings--
    step = args.restore_step + 1
    epoch = 1
    total_step = 44000
    log_step = 100
    val_step = 1000
    save_step = 2000
    grad_acc_step = 1
    grad_clip_thresh = 1.0
    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()
    
    # log setting 
    log_path = os.path.join(args.log_path, args.version)
    os.makedirs(log_path, exist_ok=True)
    
    train = True
    while train:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batch in loader:
            
            if train == False: break
            batch = to_device(batch, device)
            with amp.autocast(args.use_amp):
                output = model(*batch)
                
                losses, lang_accuracy, spker_accuracy = Loss(batch[-2:], output)
                total_loss = losses[0]
                # **mix** 
                ''' 
                total_loss, spker_accuracy = Loss(output, batch[-2])
                total_loss = total_loss / grad_acc_step
                '''
            # Backward
            scaler.scale(total_loss).backward()
            
            # Clipping gradients to avoid gradient explosion
            if step % grad_acc_step == 0:
                scaler.unscale_(optimizer._optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
            
            # Update weights         
            optimizer.step_and_update_lr(scaler)
            scaler.update()
            optimizer.zero_grad()
            
            if step % log_step == 0:
                message1 = "Step {}/{}, ".format(step, total_step)
                
                losses = [l.item() for l in losses]
                message2 = "Total Loss: {:.4f}, Spker Loss: {:.4f}, Lang Loss {:.4f}".format(*losses)
                message3 = ", Spker Accuracy: {:.4f}, Lang Accuracy: {:.4f}".format(spker_accuracy, lang_accuracy)
                '''
                message2 = "Total Loss: {:.4f}".format(total_loss)     
                message3 = ", Spker Accuracy: {:.4f}".format(spker_accuracy)
                '''
                outer_bar.write(message1 + message2 + message3)
                
                with open(os.path.join(log_path, "train_log.txt"), "a") as f:
                    f.write(message1 + message2 + message3 + "\n")
                
            if step % val_step == 0:
                model.eval()
                #message = evaluate(device, model, step, args, len(losses))
                message = evaluate(device, model, step, args)
                outer_bar.write(message)
                model.train()
                
                with open(os.path.join(log_path, "val_log.txt"), "a") as f:
                    f.write(message + "\n")
                
            if step % save_step == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer._optimizer.state_dict(),
                    },
                    os.path.join(log_path, "{}.pth.tar".format(step),),
                )
                
            # batch level update                
            if step == total_step:
                train = False
                break
            
            step += 1
            outer_bar.update(1)
            inner_bar.update(1)      
        # epoch level update       
            
        epoch += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=['LibriTTS', 'AISHELL3'],
        help="name of datasets",
    )
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--version", type=str, default="base")
    parser.add_argument("--max_wav_length", type=int, default=16000) # 3s:48000, 6s:96000
    parser.add_argument("--wavlm_pretrain_model", type=str, default="./wavlm-base-plus")
    parser.add_argument("--fix_wavLM", action='store_true')

    parser.add_argument("--preprocess_path", type=str, default="./data/mix_data/train")

    args = parser.parse_args()
    train(args)
    
    # STEP1: "Fix wavLM", Train TDNN with (3s wav, 0.2 margin) for 20 epoch
    # STEP2: Finetune whole model with (3s wav, 0.2 margin) for 5 epoch
    # STEP3: additional (6s wav, 0.4 margin) for 2 epoch