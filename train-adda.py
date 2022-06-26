from transformers import Wav2Vec2FeatureExtractor
import torch
import os 
import librosa
import numpy as np
from tqdm import tqdm

import argparse
from utils import get_model_ADDA, to_device_ADDA
from dataset import Dataset_ADDA
from torch.utils.data import DataLoader

from torch.cuda import amp
import torch.nn as nn
from evaluate import evaluate_ADDA

def train(args):
    device = torch.device('cuda')
    
    # Get dataset
    dataset = Dataset_ADDA("src_embed_filepath.txt", "train.txt", args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    
    # Prepare model
    model, D, opt_G, opt_D = get_model_ADDA(args, device, train=True)
    
    scaler = amp.GradScaler()
    criterion = nn.BCELoss()
    
    # --settings--
    step = args.restore_step + 1
    epoch = 1
    total_step = 90000
    log_step = 100
    val_step = 1000
    save_step = 20000
    grad_acc_step = 4
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
            
            ''' Train Discriminator '''           
            batch = to_device_ADDA(batch, device)

            src_embed = batch[2]
            tgt_embed = model(*batch[:2]).embeddings
            
            # Create label   
            b_s = len(batch[0])
            src_label = torch.ones((b_s)).cuda()
            tgt_label = torch.zeros((b_s)).cuda()
                  
            src_logit = D(src_embed.detach())
            tgt_logit = D(tgt_embed.detach())
       
            # compute loss
            src_loss = criterion(src_logit, src_label)
            tgt_loss = criterion(tgt_logit, tgt_label)
            loss_D = (src_loss + tgt_loss) / 2 / grad_acc_step
            
            #print(model.feature_extractor.weight.grad)
            #print(D.model[0].weight.grad)
            
            loss_D.backward()

            # Update D
            if step % grad_acc_step == 0:
                opt_D.step()
                D.zero_grad()


                
            ''' Train Generator '''
            tgt_embed = model(*batch[:2]).embeddings
            tgt_logit = D(tgt_embed)
            
            # compute loss
            loss_G = criterion(tgt_logit, src_label) / grad_acc_step
         
            for param in D.parameters():
                param.requires_grad = False
                
            loss_G.backward()
            
            for param in D.parameters():
                param.requires_grad = True
                
            # update G
            if step % grad_acc_step == 0:
                opt_G.step()   
                model.zero_grad()

               
            if step % log_step == 0:
                message1 = "Step {}/{}, ".format(step, total_step)
                message2 = "G Loss: {:.4f}, D Loss: {:.4f}".format(loss_G.item(), loss_D.item())
                outer_bar.write(message1 + message2)
                '''
                with open(os.path.join(log_path, "train_log.txt"), "a") as f:
                    f.write(message1 + message2 + message3 + "\n")
                '''
            
            if step % val_step == 0:
                model.eval()
                message = evaluate_ADDA(device, model, step, args)
                outer_bar.write(message)
                model.train()
               
                with open(os.path.join(log_path, "val_log.txt"), "a") as f:
                    f.write(message + "\n")
             
          
            if step % save_step == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--raw_path", type=str, default="/home/bai/data/local/VAE-TTS/raw_data")
    parser.add_argument("--log_path", type=str, default="./log-adda")
    parser.add_argument("--preprocess_path", type=str, default="./info-adda")
    parser.add_argument("--wav_path", type=str, default="./info/wav")

    parser.add_argument("--version", type=str, default="base")
    parser.add_argument("--max_wav_length", type=int, default=48000) # 3s:48000, 6s:96000


    args = parser.parse_args()
    train(args)