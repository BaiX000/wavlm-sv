import os
import json

import torch
import torch.nn as nn
from transformers import WavLMModel, WavLMForXVector
from gradient_reversal import grad_reverse



class WavLMSV(nn.Module):
    
    def __init__(self, args):
        super(WavLMSV, self).__init__()
        with open(os.path.join(args.preprocess_path, "speakers.json"), 'r') as f:
            speaker_map = json.load(f)    
        self.n_speaker = len(speaker_map)
        
        # wavlm settings 
        self.wavlm = WavLMModel.from_pretrained(args.wavlm_pretrain_model)
          
        # sv settings
        num_layers = 12 + 1
        self.use_weighted_layer_sum = True
        if self.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(768, 512)

        tdnn_layers = [TDNNLayer(i) for i in range(5)]
        self.tdnn = nn.ModuleList(tdnn_layers)
        self.feature_extractor = nn.Linear(1500 * 2, 512)
        
        # not USED
        #self.classifier = nn.Linear(512, 512)

        
        # select lang clsfir
        #self.lang_clsfir = LangClassifier(768, 2)
        self.lang_clsfir = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 2))
        
        # select spker clsfir
        #self.spker_clsfir = nn.Linear(512, self.n_speaker)
        self.spker_clsfir = SpkerClassifier(512, self.n_speaker)
        
    def forward(self, wavs, attn_masks, spkers=None, langs=None):

        wavlm_output = self.wavlm(
            input_values = wavs,
            attention_mask = attn_masks,
            output_hidden_states = True,
            #output_attentions=True
        )

        last_hidden_state = wavlm_output.last_hidden_state
        hidden_states = wavlm_output.hidden_states

        # sv task 
        if self.use_weighted_layer_sum:
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = last_hidden_state
        
        hidden_states = self.projector(hidden_states)
        
        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)
        
         # Statistic Pooling
        if attn_masks is None:
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attn_masks.sum(dim=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)
        
        xvector = self.feature_extractor(statistic_pooling)
        
        # SpkerClsfir: 1. normal Softmax 2.AMSoftmax
        #spker_clsfir_output = self.spker_clsfir(xvector)
        spker_clsfir_output = None
        if spkers is not None:
            spker_clsfir_output = self.spker_clsfir(xvector, spkers)
            
        # Lang Clsfir: 1.input wavLM last hidden 2.input XVector
        '''
        last_hidden_for_lang_clsfir = grad_reverse(last_hidden_state)
        lang_clsfir_output = self.lang_clsfir(last_hidden_for_lang_clsfir, feat_extract_output_lengths)   
        '''
        
        xvector_for_lang_clsfir = grad_reverse(xvector)
        lang_clsfir_output = self.lang_clsfir(xvector_for_lang_clsfir)
        
        return (spker_clsfir_output, lang_clsfir_output, xvector) 
    
    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """
        conv_kernel = [10, 3, 3, 3, 3, 2, 2]
        conv_stride = [5, 2, 2, 2, 2, 2, 2]
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(conv_kernel, conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths
    
    def _get_tdnn_output_lengths(self, input_lengths):
        """
        Computes the output length of the TDNN layers
        """
        tdnn_kernel = [5, 3, 3, 1, 1]
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size in tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths
    
    def init_weights(self):
        """
        If needed prunes and maybe initializes weights.
        """
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights"""
        # gumbel softmax requires special init
        '''
        if isinstance(module, WavLMGumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)
        elif isinstance(module, WavLMPositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, WavLMFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        '''
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            print("!!")

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        
    
class TDNNLayer(nn.Module):
    def __init__(self, layer_id=0):
        super().__init__()
        tdnn_dim = [512, 512, 512, 512, 1500]
        tdnn_kernel = [5, 3, 3, 1, 1]
        tdnn_dilation = [1, 2, 3, 1, 1]
        self.in_conv_dim = tdnn_dim[layer_id - 1] if layer_id > 0 else tdnn_dim[layer_id]
        self.out_conv_dim = tdnn_dim[layer_id]
        self.kernel_size = tdnn_kernel[layer_id]
        self.dilation = tdnn_dilation[layer_id]

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = nn.functional.unfold(
            hidden_states,
            (self.kernel_size, self.in_conv_dim),
            stride=(1, self.in_conv_dim),
            dilation=(self.dilation, 1),
        )
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.kernel(hidden_states)

        hidden_states = self.activation(hidden_states)
        return hidden_states

        
        
    
class LangClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LangClassifier, self).__init__()
        #self.model = nn.Linear(input_dim, output_dim)
        self.model = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Linear(input_dim, output_dim))
    def forward(self, last_hidden, src_len):
        '''
        input:-
            last_hidden: [B, T, D]
            src_len: [B]
        output:-
            out: [B, T, n_lang]
        '''
        out = self.model(last_hidden)
        # create mask
        batch_size, max_len = out.shape[0], out.shape[1]
        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(src_len.device)
        mask = ids >= src_len.unsqueeze(1).expand(-1, max_len)
        out = out.masked_fill(mask.unsqueeze(-1), 0)
        return out
        
class SpkerClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4, top_k=6, penalty=0.1):
        super(SpkerClassifier, self).__init__()
        self.scale = scale
        self.margin = margin
        self.top_k = top_k
        self.penalty = penalty
        self.num_labels = num_labels
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        self.classifier = nn.Linear(input_dim, input_dim)
        
    def forward(self, hidden_states, labels):
      
        hidden_states = self.classifier(hidden_states)          

        labels = labels.flatten()
        weight = nn.functional.normalize(self.weight, dim=0)
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        cos_theta = torch.mm(hidden_states, weight) # (B, 512) x (512, n_spker)
              
        psi = cos_theta - self.margin        
        onehot = nn.functional.one_hot(labels, self.num_labels)
       
        # Add penalty margin to top-K closed classes
        pen = cos_theta + self.penalty
        top_indice = torch.topk(cos_theta, self.top_k).indices
        pen_onehot = torch.zeros_like(cos_theta).scatter_(dim=-1, index=top_indice, value=1)  
        cos_theta_with_pen = torch.where(pen_onehot.bool(), pen, cos_theta)
        
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta_with_pen)
        return logits
    

    
class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, hidden_states, labels):
        labels = labels.flatten()
        weight = nn.functional.normalize(self.weight, dim=0)
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        cos_theta = torch.mm(hidden_states, weight)
        psi = cos_theta - self.margin

        onehot = nn.functional.one_hot(labels, self.num_labels)
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        loss = self.loss(logits, labels)

        return loss    
    
class WavLMExtend(nn.Module):
    def __init__(self, args):
        super(WavLMExtend, self).__init__()
        
        with open(os.path.join(args.preprocess_path, "speakers.json"), 'r') as f:
            speaker_map = json.load(f)  
        
        self.n_speaker = len(speaker_map)
        self.wavlmxvector = WavLMForXVector.from_pretrained("wavlm-base-plus-sv")
        self.wavlmxvector.objective = AMSoftmaxLoss(512, self.n_speaker)

    def forward(self, wavs, attn_masks, spkers=None, langs=None):

        output = self.wavlmxvector(
            input_values = wavs,
            attention_mask = attn_masks,
            labels = spkers,
        )

        return output
    
class TDNN(nn.Module):
    def __init__(self, args):
        super(TDNN, self).__init__()
        with open(os.path.join(args.preprocess_path, "speakers.json"), 'r') as f:
            speaker_map = json.load(f)    
        self.n_speaker = len(speaker_map)
        
        # wavlm settings 
          
        # sv settings

        self.projector = nn.Linear(20, 512)

        tdnn_layers = [TDNNLayer(i) for i in range(5)]
        self.tdnn = nn.ModuleList(tdnn_layers)
        self.feature_extractor = nn.Linear(1500 * 2, 512)
        self.spker_clsfir = SpkerClassifier(512, self.n_speaker)
        self.lang_clsfir = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 2))
    def forward(self, mfcc, src_len, spkers=None, langs=None):


        hidden_states = self.projector(mfcc)

        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)
        
        
         # Statistic Pooling
        if src_len is None:
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
            tdnn_output_lengths = self._get_tdnn_output_lengths(src_len)
            mean_features = []
            std_features = []
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)
        
        xvector = self.feature_extractor(statistic_pooling)
        # SpkerClsfir: 1. normal Softmax 2.AMSoftmax
        #spker_clsfir_output = self.spker_clsfir(xvector)
        spker_clsfir_output = None
        if spkers is not None:
            spker_clsfir_output = self.spker_clsfir(xvector, spkers)
            
        # Lang Clsfir: 1.input wavLM last hidden 2.input XVector
        '''
        last_hidden_for_lang_clsfir = grad_reverse(last_hidden_state)
        lang_clsfir_output = self.lang_clsfir(last_hidden_for_lang_clsfir, feat_extract_output_lengths)   
        '''
        
        lang_clsfir_output = self.lang_clsfir(xvector)
        
        return (spker_clsfir_output, lang_clsfir_output, xvector)
    
    def _get_tdnn_output_lengths(self, input_lengths):
        """
        Computes the output length of the TDNN layers
        """
        tdnn_kernel = [5, 3, 3, 1, 1]
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size in tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths
'''
class ADDA(nn.Module):
    def __init__(self, args):
        super(WavLMSV, self).__init__()
        self.wavlmsv = WavLMForXVector.from_pretrained("wavlm-base-plus-sv")

    def forward(self, wavs, attn_masks, spkers=None, langs=None):
        output = self.wavlmsv(wavs, attn_masks)
        domain = self.domain_clsfir(output.embeddings)
        print(output, domain)
        return output, domain

class DomainClsfir(nn.Module):
    def __init__(self):
        super(DomainClsfir, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1), 
            nn.Sigmoid()
        )
        
    def forward(self, embed):
        output = self.model(embed)
        output = output.view(-1)
        return output
'''