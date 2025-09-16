import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


    

class SpatialAttention_new(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_new, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(4, 1, kernel_size, padding=padding, bias=False)
        self.conv1_help = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv1_y = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,help):
        avg_out_x1 = torch.mean(x, dim=1, keepdim=True)
        max_out_x1, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out_x1, max_out_x1], dim=1)


        avg_out = torch.mean(help, dim=1, keepdim=True)
        max_out, _ = torch.max(help, dim=1, keepdim=True)
        help = torch.cat([avg_out, max_out], dim=1)
        help = self.conv1_help(help)
        sig_help = self.sigmoid(help)

        avg_out_x2 = avg_out_x1*sig_help
        max_out_x2 = max_out_x1*sig_help
        x = torch.cat([avg_out_x2 , max_out_x2 ], dim=1)
        x = torch.cat([x , y ], dim=1)
        x = self.conv1(x)
        
        # x=x+y
        sig_x = self.sigmoid(x)
        sig_help = self.sigmoid(sig_help+sig_x)
        return sig_x,sig_help




class CSIAM(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CSIAM, self).__init__()
        self.xchannelattention = ChannelAttention(channel, ratio=ratio)
        self.helpchannelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention_new(kernel_size=kernel_size)

    def forward(self, x,help):
        x = x*self.xchannelattention(x) 
        help = help*self.helpchannelattention(help) 

        y_x=x
        y_hekp=help
        
        sig_x,sig_help= self.spatialattention(x,help)
        x=x*sig_x + y_x
        help=help*sig_help + y_hekp

        return x , help

    
