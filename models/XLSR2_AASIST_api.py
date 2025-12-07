import logging
import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import fairseq


___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

############################
## FOR fine-tuned SSL MODEL
############################


class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        
        cp_path = '/work/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/models/weights/xlsr2_300m.pt'   # Change the pre-trained XLSR model path.
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        
        # put the model to GPU if it not there
        # if next(self.model.parameters()).device != input_data.device \
        #    or next(self.model.parameters()).dtype != input_data.dtype:
        #     self.model.to(input_data.device, dtype=input_data.dtype)

        if True:
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data

            # [batch, length, dim]
            out_dict = self.model(input_tmp, mask=False, features_only=True)
            emb = out_dict['x']
            layerresult = out_dict['layer_results']
        return emb, layerresult




class LayerSelfAttention(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                               num_heads=num_heads,
                                               batch_first=True)  # batch在前

    def forward(self, layerResult):
        """
        layery: (b, L, d)   -> (batch, seq_len, hidden_dim)
        """
        alllayerResult = []

        for layer in layerResult:
            layery = layer[0].transpose(0, 1)  # (x,z)  x(201,b,1024) (b,201,1024)
            alllayerResult.append(layery)

        layery = torch.cat(alllayerResult, dim=1)

        att_out, att_weights = self.self_attn(layery, layery, layery)
        # att_out: (b, L, d)
        # att_weights: (b, L, L) 每个 token(层)对其它层的注意力权重

        # 可以再加一层池化，得到最终全局表示
        pooled = att_out.mean(dim=1)   # (b, d)

        return att_out, pooled, att_weights



class Model(nn.Module):
    def __init__(self, args,device='cuda'):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        self.layer_att = LayerSelfAttention(hidden_dim=self.ssl_model.out_dim)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(self.ssl_model.out_dim, 128)
        self.fc3 = nn.Linear(128,21)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, Freq_aug=False):

        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)

        att_out, pooled, _ = self.layer_att(layerResult)

        x = self.fc1(pooled)
        x = self.selu(x)
        x = self.fc3(x)
        x = self.selu(x)
        output = self.logsoftmax(x)
        dummy = torch.zeros(x.size(0), device=x.device)  # 与 batch 对齐
        return pooled, x


