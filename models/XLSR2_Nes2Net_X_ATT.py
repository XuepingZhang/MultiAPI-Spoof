
import fairseq

import torch
import torch.nn as nn
import math

class LocalWindowAttention(nn.Module):
    """
    Local window (sliding) cross-block attention.
    Input: xb [B, NB, C, T]
    Output: xb' [B, NB, C, T]  (residual fused)
    Params:
      - window_size: odd integer W (number of blocks to attend for each center)
      - dilation: step between neighbor blocks (default 1)
      - num_heads: multi-head attention heads (internal MHA implemented via linear projections)
      - reduce_ratio: linear proj reduce factor (embed_dim = max(C//reduce_ratio, 1))
    Boundary handling:
      - uses clamped indices (edge repeats). Could be changed to zero-pad if desired.
    """
    def __init__(self, num_blocks, block_channels, window_size=3, dilation=1,
                 num_heads=4, reduce_ratio=1, dropout=0.0, use_pos=True):
        super().__init__()
        assert window_size % 2 == 1, "window_size must be odd"
        self.NB = num_blocks
        self.C = block_channels
        self.W = window_size
        self.dilation = dilation
        self.num_heads = num_heads
        self.use_pos = use_pos

        D = max(block_channels // reduce_ratio, 1)
        # ensure D divisible by num_heads for simple multi-head split
        head_dim = max(D // num_heads, 1)
        D = head_dim * num_heads
        self.D = D
        self.head_dim = head_dim

        self.proj_q = nn.Linear(block_channels, D, bias=False)
        self.proj_k = nn.Linear(block_channels, D, bias=False)
        self.proj_v = nn.Linear(block_channels, D, bias=False)
        self.proj_out = nn.Linear(D, block_channels, bias=False)

        self.dropout = nn.Dropout(dropout)
        if use_pos:
            # position embedding per block (on D dims)
            self.pos_emb = nn.Parameter(torch.zeros(self.NB, D))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        else:
            self.pos_emb = None

    def _build_neighbor_index(self, device):
        """
        Precompute neighbor index matrix of shape [NB, W] where each row i contains the
        indices (clamped) of the window centered at i with given dilation.
        """
        half = (self.W - 1) // 2
        idx = torch.arange(self.NB, device=device)  # [NB]
        # neighbors offsets: [-half..half] * dilation
        offsets = torch.arange(-half, half+1, device=device) * self.dilation  # [W]
        neighbors = idx.unsqueeze(1) + offsets.unsqueeze(0)  # [NB, W]
        neighbors = torch.clamp(neighbors, 0, self.NB - 1)   # clamp to edges (repeat edges)
        return neighbors.long()  # [NB, W]

    def forward(self, xb):
        # xb: [B, NB, C, T]
        B, NB, C, T = xb.shape
        assert NB == self.NB and C == self.C

        block_tokens = xb.mean(dim=-1)           # [B, NB, C]

        Q = self.proj_q(block_tokens)   # [B, NB, D]
        K = self.proj_k(block_tokens)   # [B, NB, D]
        V = self.proj_v(block_tokens)   # [B, NB, D]
        if self.pos_emb is not None:

            pos = self.pos_emb.unsqueeze(0)  # [1, NB, D]
            Q = Q + pos
            K = K + pos
            V = V + pos

        neighbors_idx = self._build_neighbor_index(device=xb.device)  # [NB, W]
        K_neighbors = K.unsqueeze(1).expand(B, NB, NB, self.D).gather(
            2, neighbors_idx.unsqueeze(0).unsqueeze(-1).expand(B, NB, self.W, self.D)
        )
        V_neighbors = V.unsqueeze(1).expand(B, NB, NB, self.D).gather(
            2, neighbors_idx.unsqueeze(0).unsqueeze(-1).expand(B, NB, self.W, self.D)
        )
        Qc = Q.unsqueeze(2)  # [B, NB, 1, D]

        scores = torch.matmul(Qc, K_neighbors.transpose(-2, -1)).squeeze(2) # [B, NB, W]
        scores = scores / math.sqrt(self.head_dim * self.num_heads)  # scale
        attn = torch.softmax(scores, dim=-1)  # [B, NB, W]
        attn = self.dropout(attn)
        attn_exp = attn.unsqueeze(2)

        context = torch.matmul(attn_exp, V_neighbors).squeeze(2) # [B, NB, D]

        # 6) project out and fuse
        out_tokens = self.proj_out(context)  # [B, NB, C]
        # expand to time and residual add
        out_tokens_time = out_tokens.unsqueeze(-1).expand(-1, -1, -1, T)  # [B, NB, C, T]
        out = xb + out_tokens_time  # residual fusion

        return out



class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        # cp_path = '/public/home/qinxy/zxp/new_dataset_experiment/aasist-main/models/weights/xlsr2_300m.pt'  # Change the pre-trained XLSR model path.
        cp_path = '/work/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/models/weights/xlsr2_300m.pt'  # Change the pre-trained XLSR model path.

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()
        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb

class SEModule(nn.Module):
    def __init__(self, channels, SE_ratio=8):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // SE_ratio, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(channels // SE_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8, SE_ratio=8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        weighted_sum = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=(kernel_size,1), dilation=(dilation, 1), padding=(num_pad, 0)))
            bns.append(nn.BatchNorm2d(width))
            initial_value = torch.ones(1, 1, 1, i+2) * (1 / (i+2))
            weighted_sum.append(nn.Parameter(initial_value, requires_grad=True))
        self.weighted_sum = nn.ParameterList(weighted_sum)
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se = SEModule(planes,SE_ratio)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out).unsqueeze(-1)  # bz c T 1

        spx = torch.split(out, self.width, 1)
        sp = spx[self.nums]
        for i in range(self.nums):
          sp = torch.cat((sp, spx[i]), -1)

          sp = self.bns[i](self.relu(self.convs[i](sp)))
          sp_s = sp * self.weighted_sum[i]
          sp_s = torch.sum(sp_s, dim=-1, keepdim=False)

          if i==0:
            out = sp_s
          else:
            out = torch.cat((out, sp_s), 1)
        out = torch.cat((out, spx[self.nums].squeeze(-1)),1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out = self.dropout(out)
        out += residual
        return out 

class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """
    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False):
        super(ASTP, self).__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim,
                                 kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(
            self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-10))
        return torch.cat([mean, std], dim=1)

class Nested_Res2Net_TDNN(nn.Module):

    def __init__(self, Nes_ratio=[8, 8], input_channel=1024, n_output_logits=2, dilation=2, pool_func='mean', SE_ratio=[8]):

        super(Nested_Res2Net_TDNN, self).__init__()
        self.Nes_ratio = Nes_ratio[0]
        assert input_channel % Nes_ratio[0] == 0
        C = input_channel // Nes_ratio[0]
        self.C = C
        Build_in_Res2Nets = []
        bns = []
        for i in range(Nes_ratio[0]-1):
            Build_in_Res2Nets.append(Bottle2neck(C, C, kernel_size=3, dilation=dilation, scale=Nes_ratio[1], SE_ratio=SE_ratio[0]))
            bns.append(nn.BatchNorm1d(C))
        self.Build_in_Res2Nets  = nn.ModuleList(Build_in_Res2Nets)
        self.bns  = nn.ModuleList(bns)
        self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.pool_func = pool_func
        if pool_func == 'mean':
            self.fc = nn.Linear(1024, n_output_logits)
        elif pool_func == 'ASTP':
            self.pooling = ASTP(in_dim=input_channel, bottleneck_dim=128, global_context_att=False)
            self.fc = nn.Linear(2048, n_output_logits)

        # 假设你已有 self.Nes_ratio, self.C 初始化
        self.local_block_attn = LocalWindowAttention(num_blocks=self.Nes_ratio,
                                                 block_channels=self.C,
                                                 window_size=3,  # W=3 -> center + 1 neighbor each side
                                                 dilation=1,
                                                 num_heads=1,
                                                 # we implemented single linear-proj MHA-like; num_heads kept as param for scaling decisions
                                                 reduce_ratio=1,
                                                 dropout=0.1,
                                                 use_pos=True)
        self.global_attn = nn.MultiheadAttention(embed_dim=input_channel, num_heads=8, batch_first=True)

    def forward(self, x):
        residual = x

        spx = torch.split(x, self.C, 1)  # tuple of [B, C, T]
        sp_list = []
        for i in range(self.Nes_ratio - 1):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.Build_in_Res2Nets[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            sp_list.append(sp)
        sp_list.append(spx[-1])  # last block unchanged per original logic

        # stack to [B, NB, C, T]
        xb = torch.stack(sp_list, dim=1)

        # apply local-window cross-block attention
        xb = self.local_block_attn(xb)

        # concat back to channel dimension: [B, NB*C, T]
        out = torch.cat([xb[:, j] for j in range(self.Nes_ratio)], dim=1)

        out = self.bn(out)
        out = self.relu(out)
        if self.pool_func == 'mean':
            out = torch.mean(out, dim=-1)
        elif self.pool_func == 'ASTP':
            out = self.pooling(out)

        out = self.fc(out)
        return out


class Model(nn.Module):
    def __init__(self, args,device='cuda'):
        super().__init__()
        self.device = device
        
        self.n_output_logits = args["n_output_logits"]

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.Nested_Res2Net_TDNN = Nested_Res2Net_TDNN(Nes_ratio=args["Nes_ratio"],
                                                       input_channel=1024,
                                                       n_output_logits=self.n_output_logits,
                                                       dilation=args["dilation"],
                                                       pool_func=args["pool_func"],
                                                       SE_ratio=args["SE_ratio"])
    def forward(self, x, Freq_aug=False):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x_ssl_feat = x_ssl_feat.permute(0,2,1)
        output = self.Nested_Res2Net_TDNN(x_ssl_feat)

        return x_ssl_feat, output

