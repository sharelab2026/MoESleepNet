from MambaSleepNet import MambaSleepNet_MultiDCT_moe
from MambaSleepNet import MambaSleepNet_MultiDCT_moe_woAu
from Attnsleep import AttnSleep_moe
import math
import torch
from torch import nn
from torch.autograd import Variable
from args_WUU import Path, Config
from simba import Block_mamba, Block_mamba_dct
# from simba import Block_mamba_dct as Block_mamba
import numpy as np


class MoESleepNet(nn.Module):
    
    def __init__(self, config):
        super(MoESleepNet, self).__init__()

        self.config = config
        self.rawNet = AttnSleep_moe()
        self.psdNet = nn.Linear(51, 1024)
        self.TFNet = MambaSleepNet_MultiDCT_moe(self.config)

        self.gating_Net = nn.Linear(3*1024, 3)
        self.k =2

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5))
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        b, v, l = x.shape
        x_raw = x[:, 0, :3000].view(b, 1, -1)
        x_psd = x[:, 1, :51].view(b, 1, -1)
        x_TF = x[:, 2, :].view(b, 1, 29, 128)

        x_raw_feature = self.rawNet(x_raw)
        x_psd_feature = self.psdNet(x_psd.view(b, -1))
        x_TF_feature = self.TFNet(x_TF)

        features = torch.cat([x_raw_feature, x_psd_feature, x_TF_feature], dim=-1)
        gating_weights = torch.softmax(self.gating_Net(features),dim=-1)
        topk_weights, topk_indices = torch.topk(gating_weights, self.k, dim=-1)

        expert_features = torch.stack([x_raw_feature, x_psd_feature, x_TF_feature], dim=1)
        final_output = torch.sum(topk_weights.unsqueeze(-1) * torch.gather(expert_features, 1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_features.size(-1))), dim=1)
        output_mid = self.fc1(final_output)
        output = self.fc2(output_mid)
        return output
    
    def get_middle_feature(self, x):
        b, v, l = x.shape
        x_raw = x[:, 0, :3000].view(b, 1, -1)
        x_psd = x[:, 1, :51].view(b, 1, -1)
        x_TF = x[:, 2, :].view(b, 1, 29, 128)

        x_raw_feature = self.rawNet(x_raw)
        x_psd_feature = self.psdNet(x_psd.view(b, -1))
        x_TF_feature = self.TFNet(x_TF)

        features = torch.cat([x_raw_feature, x_psd_feature, x_TF_feature], dim=-1)
        gating_weights = torch.softmax(self.gating_Net(features),dim=-1)
        topk_weights, topk_indices = torch.topk(gating_weights, self.k, dim=-1)
        

        expert_features = torch.stack([x_raw_feature, x_psd_feature, x_TF_feature], dim=1)
        final_output = torch.sum(topk_weights.unsqueeze(-1) * torch.gather(expert_features, 1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_features.size(-1))), dim=1)
        output_mid = self.fc1(final_output)
        output = self.fc2(output_mid)
        
        return (x_raw_feature.view(x.size(0), -1), x_psd_feature.view(x.size(0), -1), x_TF_feature.view(x.size(0), -1),\
                 final_output.view(x.size(0), -1), output_mid)
    
    def get_topk_indices(self, x):
        b, v, l = x.shape
        x_raw = x[:, 0, :3000].view(b, 1, -1)
        x_psd = x[:, 1, :51].view(b, 1, -1)
        x_TF = x[:, 2, :].view(b, 1, 29, 128)

        x_raw_feature = self.rawNet(x_raw)
        x_psd_feature = self.psdNet(x_psd.view(b, -1))
        x_TF_feature = self.TFNet(x_TF)

        features = torch.cat([x_raw_feature, x_psd_feature, x_TF_feature], dim=-1)
        gating_weights = torch.softmax(self.gating_Net(features),dim=-1)
        topk_weights, topk_indices = torch.topk(gating_weights, self.k, dim=-1)
        
        return topk_indices
    
class MoESleepNet_woMerge(nn.Module):
    
    def __init__(self, config):
        super(MoESleepNet_woMerge, self).__init__()

        self.config = config
        self.rawNet = AttnSleep_moe()
        self.psdNet = nn.Linear(51, 1024)
        self.TFNet = MambaSleepNet_MultiDCT_moe(self.config)

        self.gating_Net = nn.Linear(3*1024, 3)
        self.k =2

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5))
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        b, v, l = x.shape
        x_raw = x[:, 0, :3000].view(b, 1, -1)
        x_psd = x[:, 1, :51].view(b, 1, -1)
        x_TF = x[:, 2, :].view(b, 1, 29, 128)

        x_raw_feature = self.rawNet(x_raw)
        x_psd_feature = self.psdNet(x_psd.view(b, -1))
        x_TF_feature = self.TFNet(x_TF)

        # features = torch.cat([x_raw_feature, x_psd_feature, x_TF_feature], dim=-1)
        # gating_weights = torch.softmax(self.gating_Net(features),dim=-1)
        # topk_weights, topk_indices = torch.topk(gating_weights, self.k, dim=-1)

        # expert_features = torch.stack([x_raw_feature, x_psd_feature, x_TF_feature], dim=1)
        # final_output = torch.sum(topk_weights.unsqueeze(-1) * torch.gather(expert_features, 1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_features.size(-1))), dim=1)
        
        final_output = x_raw_feature + x_psd_feature + x_TF_feature
        output_mid = self.fc1(final_output)
        output = self.fc2(output_mid)
        return output

class MoESleepNet_woMerge2(nn.Module):
    
    def __init__(self, config):
        super(MoESleepNet_woMerge2, self).__init__()

        self.config = config
        self.rawNet = AttnSleep_moe()
        self.psdNet = nn.Linear(51, 1024)
        self.TFNet = MambaSleepNet_MultiDCT_moe(self.config)

        self.gating_Net = nn.Linear(3*1024, 3)
        self.k =2

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5))
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        b, v, l = x.shape
        x_raw = x[:, 0, :3000].view(b, 1, -1)
        x_psd = x[:, 1, :51].view(b, 1, -1)
        x_TF = x[:, 2, :].view(b, 1, 29, 128)

        x_raw_feature = self.rawNet(x_raw)
        x_psd_feature = self.psdNet(x_psd.view(b, -1))
        x_TF_feature = self.TFNet(x_TF)

        # features = torch.cat([x_raw_feature, x_psd_feature, x_TF_feature], dim=-1)
        # gating_weights = torch.softmax(self.gating_Net(features),dim=-1)
        # topk_weights, topk_indices = torch.topk(gating_weights, self.k, dim=-1)

        # expert_features = torch.stack([x_raw_feature, x_psd_feature, x_TF_feature], dim=1)
        # final_output = torch.sum(topk_weights.unsqueeze(-1) * torch.gather(expert_features, 1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_features.size(-1))), dim=1)
        
        final_output = x_raw_feature + x_psd_feature
        output_mid = self.fc1(final_output)
        output = self.fc2(output_mid)
        return output

class MoESleepNet_woAu(nn.Module):
    
    def __init__(self, config):
        super(MoESleepNet_woAu, self).__init__()

        self.config = config
        self.rawNet = AttnSleep_moe()
        self.psdNet = nn.Linear(51, 1024)
        self.TFNet = MambaSleepNet_MultiDCT_moe_woAu(self.config)

        self.gating_Net = nn.Linear(3*1024, 3)
        self.k =2

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5))
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        b, v, l = x.shape
        x_raw = x[:, 0, :3000].view(b, 1, -1)
        x_psd = x[:, 1, :51].view(b, 1, -1)
        x_TF = x[:, 2, :].view(b, 1, 29, 128)

        x_raw_feature = self.rawNet(x_raw)
        x_psd_feature = self.psdNet(x_psd.view(b, -1))
        x_TF_feature = self.TFNet(x_TF)

        features = torch.cat([x_raw_feature, x_psd_feature, x_TF_feature], dim=-1)
        gating_weights = torch.softmax(self.gating_Net(features),dim=-1)
        topk_weights, topk_indices = torch.topk(gating_weights, self.k, dim=-1)

        expert_features = torch.stack([x_raw_feature, x_psd_feature, x_TF_feature], dim=1)
        final_output = torch.sum(topk_weights.unsqueeze(-1) * torch.gather(expert_features, 1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_features.size(-1))), dim=1)
        output_mid = self.fc1(final_output)
        output = self.fc2(output_mid)
        return output


if __name__ == '__main__':
    
    config = Config()
    x = torch.rand((64, 3, 3712)).to(config.device)
    m = MoESleepNet(config).to(config.device)
    y = m(x)
    print(y.shape)



