import torch.nn as nn
import torch
from nets.CSIAM import CSIAM



import torch.nn.functional as F

class owtDnet(nn.Module):
    def __init__(self, features, features_help,num_classes=1000):
        super(owtDnet, self).__init__()
        
        self.CSIAM3=CSIAM(256)
        self.CSIAM4=CSIAM(512)
        self.CSIAM5=CSIAM(512)

        self.features = features
        self.features_help = features_help


        self._initialize_weights()



    def forward(self, x):
        help=x[:, :10, :, :]  #sentinel1
        x = x[:, 10:, :, :]   #sentinel2  

        ws1,_=torch.max(help[:, :2, :, :], dim=1, keepdim=True)
        ws2,_=torch.max(x[:, :3, :, :], dim=1, keepdim=True)
        ws1=ws2+ws1
        

        feat1 = self.features[  :4 ](x)
        feat1_help = self.features_help[  :4 ](help)
        

        feat2 = self.features[4 :9 ](feat1)
        feat2_help = self.features_help[  4 :9 ](feat1_help)
       

        feat3 = self.features[9 :14](feat2)
        feat3_help = self.features_help[  9 :14 ](feat2_help)
        feat3,feat3_help = self.CSIAM3(feat3,feat3_help)  
        

        feat4 = self.features[14:19](feat3)
        feat4_help = self.features_help[  14:19 ](feat3_help)
        feat4,feat4_help = self.CSIAM4(feat4,feat4_help) 
        

        feat5 = self.features[19:-1](feat4)
        feat5_help = self.features_help[  19:-1 ](feat4_help)
        feat5,feat5_help = self.CSIAM5(feat5,feat5_help) 

        return [feat1, feat2, feat3, feat4, feat5,feat1_help, feat2_help, feat3_help, feat4_help, feat5_help,ws2,ws1]


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels = 3):
    layers = []
    # in_channels = 14
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
}


def owtDnetmodel(in_channels = 15, in_channels_help = 10, **kwargs):
    model = owtDnet(make_layers(cfgs["D"], batch_norm = False, in_channels = in_channels),
                make_layers(cfgs["D"], batch_norm = False, in_channels= in_channels_help),
                    **kwargs)
    return model
