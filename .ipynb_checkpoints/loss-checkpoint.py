import torch
import torch.nn as nn
import os
import json
'''
class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        self.loss = nn.CrossEntropyLoss()
        
        self.classifier = nn.Linear(512, 512)
        
    def forward(self, hidden_states, labels):
        # need? 
        hidden_states = self.classifier(hidden_states)
        
        labels = labels.flatten()
     
        weight = nn.functional.normalize(self.weight, dim=0)
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        cos_theta = torch.mm(hidden_states, weight)
        psi = cos_theta - self.margin

        onehot = nn.functional.one_hot(labels, self.num_labels)
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        
        loss = self.loss(logits, labels)

        return loss, logits
'''
class WavLMSVLoss(nn.Module):
    
    def __init__(self, args):
        super(WavLMSVLoss, self).__init__()
        
        with open(os.path.join(args.preprocess_path, "speakers.json"), 'r') as f:
            speaker_map = json.load(f)    
        self.n_speaker = len(speaker_map)
        
        #self.AMSoftmaxLoss = AMSoftmaxLoss(512, self.n_speaker)
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        n_libritts_uttr, n_aishell3_uttr = 75695, 30723
        uttr_sum = n_libritts_uttr + n_aishell3_uttr
        class_weights = [1-(n_libritts_uttr/uttr_sum), 1-(n_aishell3_uttr/uttr_sum)]
        class_weights = torch.FloatTensor(class_weights).cuda()
        self.weighted_ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.lang_weight = 0.0
        self.spker_weight = 1.0
    
        
    def forward(self, targets, predictions):
        
        spker_targets = targets[0]
        lang_targets = targets[1]
        spker_predictions = predictions[0]
        lang_predictions = predictions[1]
        xvector = predictions[2]

        spker_targets.requires_grad = False
        lang_targets.requires_grad = False    
        
        '''
        lang_predictions = [B, L, n_lang] -> (reshape) : [B*L, n_lang]
        lang_target = [B] -> expand_lang_targets = [B*L]
        '''
        
        
        # spker_loss selection
        # 1. general softmax output / 2. AMSoftmax output
        spker_loss = self.ce_loss(spker_predictions, spker_targets)
        
        # lang loss selection
        # 1. wavLM last hidden output / 2.XVector
        # lang target expansion for wavLM last hidden
        '''
        expand_lang_targets = lang_targets.repeat_interleave(lang_predictions.shape[1])
        lang_predictions = lang_predictions.reshape(-1, lang_predictions.shape[-1])
        mask_index = lang_predictions.abs().sum(dim=1) != 0
        lang_predictions = lang_predictions[mask_index]
        expand_lang_targets = expand_lang_targets[mask_index]
        '''
        #lang_loss = self.weighted_ce_loss(lang_predictions, expand_lang_targets)
        lang_loss = self.ce_loss(lang_predictions, lang_targets)

        #spker_loss = spker_loss*self.spker_weight
        spker_loss = spker_loss*self.spker_weight
        lang_loss = lang_loss*self.lang_weight
        total_loss = spker_loss + lang_loss
        
        # select lang accuracy: 1. wavLM last hidden output / 2.XVector
        #lang_accuracy = (torch.argmax(lang_predictions, dim=1) == expand_lang_targets).sum().item() / expand_lang_targets.shape[0]
        lang_accuracy = (torch.argmax(lang_predictions, dim=1) == lang_targets).sum().item() / lang_targets.shape[0]
        
        # select spker accuracy: 1. general softmax output / 2. AMSoftmax output
        spker_accuracy = (torch.argmax(spker_predictions, dim=1) == spker_targets).sum().item() / spker_targets.shape[0]
        
        return (total_loss, spker_loss, lang_loss), lang_accuracy, spker_accuracy
    

class WavLMExtendLoss(nn.Module):
    
    def __init__(self, args):
        super(WavLMExtendLoss, self).__init__()
        
        with open(os.path.join(args.preprocess_path, "speakers.json"), 'r') as f:
            speaker_map = json.load(f)    
        self.n_speaker = len(speaker_map)    
        
    def forward(self, output, spker_targets):
        
        spker_targets.requires_grad = False
          
        spker_accuracy = (torch.argmax(output.logits, dim=1) == spker_targets).sum().item() / spker_targets.shape[0]
  
        return output.loss, spker_accuracy