import math

import torch
from torch import nn
from torch.autograd import Variable
from args_WUU import Path, Config
from simba import Block_mamba, Block_mamba_dct
# from simba import Block_mamba_dct as Block_mamba



class MultiSpectralAttentionLayer1D(nn.Module):
    def __init__(self, channel, length, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer1D, self).__init__()
        self.reduction = reduction
        self.length = length


        self.mapper_top = get_freq_indices('top' + freq_sel_method[3:])
        self.mapper_bot = get_freq_indices('bot' + freq_sel_method[3:])
        self.mapper_low = get_freq_indices('low' + freq_sel_method[3:])
        self.num_split = len(self.mapper_top)
        # Adjust frequency indices for the length
        self.mapper_top = [temp * (length // 7) for temp in self.mapper_top]
        self.mapper_bot = [temp * (length // 7) for temp in self.mapper_bot]
        self.mapper_low = [temp * (length // 7) for temp in self.mapper_low]  

        self.dct_layer_top = MultiSpectralDCTLayer1D(length, self.mapper_top, channel)
        self.dct_layer_bot = MultiSpectralDCTLayer1D(length, self.mapper_bot, channel)
        self.dct_layer_low = MultiSpectralDCTLayer1D(length, self.mapper_low, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, l = x.shape
        if l != self.length:
            x_pooled = torch.nn.functional.adaptive_avg_pool1d(x, self.length)
        else:
            x_pooled = x
        y_top = self.dct_layer_top(x_pooled) # (5, 29)
        y_bot = self.dct_layer_bot(x_pooled)
        y_low = self.dct_layer_low(x_pooled)


        y_top = self.fc(y_top).view(n, c, 1) # (5, 29, 1)
        y_bot = self.fc(y_bot).view(n, c, 1)
        y_low = self.fc(y_low).view(n, c, 1)
        return x * y_top.expand_as(x) + x * y_bot.expand_as(x) + x * y_low.expand_as(x)

class MultiSpectralDCTLayer1D(nn.Module):
    def __init__(self, length, mapper, channel):
        super(MultiSpectralDCTLayer1D, self).__init__()
        assert channel % len(mapper) == 0

        self.num_freq = len(mapper)
        self.register_buffer('weight', self.get_dct_filter_1d(length, mapper, channel))

    def forward(self, x):
        assert len(x.shape) == 3, 'x must be 3 dimensions, but got ' + str(len(x.shape))
        x = x * self.weight

        result = torch.sum(x, dim=2)
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        return result * math.sqrt(2) if freq > 0 else result

    def get_dct_filter_1d(self, length, mapper, channel):
        dct_filter = torch.zeros(channel, length)
        c_part = channel // self.num_freq

        for i, u in enumerate(mapper):
            for t in range(length):
                dct_filter[i * c_part: (i + 1) * c_part, t] = self.build_filter(t, u, length)

        return dct_filter

class FcaBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16, freq_sel_method_name='bot29'):
        global _mapper_x
        super(FcaBasicBlock, self).__init__()
        # assert fea_h is not None
        # assert fea_w is not None
        c2l = dict([(29, 128), (128, 28), (128*3, 29)]) # 键值对，29代表通道，384代表特征的长度
        self.planes = planes
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.att = MultiSpectralAttentionLayer1D(planes, c2l[planes],  reduction=reduction, freq_sel_method = freq_sel_method_name)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MambaSleepNet_MultiDCT_moe(nn.Module):
    
    def __init__(self, config):
        super(MambaSleepNet_MultiDCT_moe, self).__init__()

        self.position_single = PositionalEncoding(d_model=config.dim_model, dropout=0.1)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=config.dim_model, nhead=config.num_head, dim_feedforward=config.forward_hidden, dropout=config.dropout)
        # self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)
        self.transformer_encoder_1 = nn.ModuleList([Block_mamba_dct(dim=128, mlp_ratio=0.3, drop_path=0.2, cm_type='EinFFT') for _ in range(config.num_encoder)])

        # self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)
        # self.transformer_encoder_3 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)

        self.drop = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(config.dim_model)

        self.position_multi = PositionalEncoding(d_model=config.dim_model, dropout=0.1)
        # encoder_layer_multi = nn.TransformerEncoderLayer(d_model=config.dim_model, nhead=config.num_head,dim_feedforward=config.forward_hidden, dropout=config.dropout)
        # self.transformer_encoder_multi = nn.TransformerEncoder(encoder_layer_multi, num_layers=config.num_encoder_multi)
        self.transformer_encoder_multi = nn.ModuleList([Block_mamba_dct(dim=128, mlp_ratio=0.3, drop_path=0.2, cm_type='EinFFT') for _ in range(config.num_encoder_multi)])

        self.fc1 = nn.Sequential(
            nn.Linear(config.pad_size * config.dim_model, config.fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(config.fc_hidden, config.num_classes)
        )

        self.dct_layer = FcaBasicBlock(29, 29) # 29是时间维度

        
        self.global_pool_time = nn.Linear(128,  2)
        self.global_pool_freq = nn.Linear(29, 2)

        self.fc_au = nn.Sequential(
            nn.Linear(config.pad_size * config.dim_model, config.fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(config.fc_hidden, config.num_classes_au)
        )

    def forward(self, x):
        x1 = x[:, 0, :, :]
        # x2 = x[:, 1, :, :]
        # x3 = x[:, 2, :, :]
        
        x1 = self.position_single(x1)
        # x2 = self.position_single(x2)
        # x3 = self.position_single(x3)

        for block in self.transformer_encoder_1:
            x1 = block(x1, 1, 29)     # (batch_size, 29, 128), (batch, time, frequency)
        # x2 = self.transformer_encoder_2(x2)
        # x3 = self.transformer_encoder_3(x3)

        # x = torch.cat([x1, x2, x3], dim=2)
        x = self.dct_layer(x1)

        x = self.drop(x)
        x = self.layer_norm(x)
        residual = x

        x = self.position_multi(x)

        for block in self.transformer_encoder_multi:
            x = block(x, 1, 29)

        x = self.layer_norm(x + residual)       # residual connection
        
        ## auxilary classifier By WUU
        x_time = self.global_pool_time(x)
        x_freq = x.permute(0, 2, 1)
        x_freq = self.global_pool_freq(x_freq)
        x_freq = x_freq.permute(0, 2, 1)
        x_au = torch.bmm(x_time, x_freq)
        x_au = self.drop(x_au)
        x_au = x_au.view(x_au.size(0), -1)
        x_au_residual = x_au
        # x_au = self.fc_au(x_au)
        ## auxilary classifier By WUU

        x = x.view(x.size(0), -1) + x_au_residual # 这里增加辅助分类器的特征
        x = self.fc1(x)
        # x = self.fc2(x)
        return x

