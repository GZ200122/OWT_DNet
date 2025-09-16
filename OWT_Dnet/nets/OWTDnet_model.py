import torch
import torch.nn as nn

from nets.OWTDnet_encode import owtDnetmodel



class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class OWTDnet(nn.Module):
    def __init__(self, num_classes = 2):
        super(OWTDnet, self).__init__()
        self.owtDnet    = owtDnetmodel()
        in_filters  = [192, 384, 768, 1024]

        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # upsampling 
        # 64,64,512
        self.up_concat4RH = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3RH = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2RH = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1RH = unetUp(in_filters[0], out_filters[0])


         # (512,512,64)cat
        self.up_temp1 = unetUp(in_filters[0], out_filters[0])


        self.cov1 = nn.Conv2d(out_filters[0]*3, out_filters[0],kernel_size=3,padding=1)
        self.cov1_1 = nn.Conv2d(out_filters[0], out_filters[0],kernel_size=3,padding=1)

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.final_1s2 = nn.Conv2d(out_filters[0], num_classes, 1)
        self.final_2vv = nn.Conv2d(out_filters[0], num_classes, 1)
        self.cov1_temp1 = nn.Conv2d(in_filters[0], out_filters[0],kernel_size=3,padding=1)
        self.cov1_temp2 = nn.Conv2d(in_filters[1], out_filters[1],kernel_size=3,padding=1)
        self.final_temp = nn.Conv2d(out_filters[0], num_classes, 1)
        self.relu   = nn.ReLU(inplace = True)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)



    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5,out_feat1, out_feat2, out_feat3, out_feat4, out_feat5,ws2,ws1] = self.owtDnet.forward(inputs)
        

        up4 = self.up_concat4(feat4, feat5)  #512  s2
        up3 = self.up_concat3(feat3, up4)    #256
        up2 = self.up_concat2(feat2, up3)    #128
        up1 = self.up_concat1(feat1, up2)    #64


        up4RH = self.up_concat4RH(out_feat4, out_feat5)  #s1
        up3RH = self.up_concat3RH(out_feat3, up4RH)
        up2RH = self.up_concat2RH(out_feat2, up3RH)
        up1RH = self.up_concat1RH(out_feat1, up2RH)


        #---------Auxiliary Semantic Segmentation Header (ASSH)-----
        out_feat33=self.up(out_feat3)
        out_feat22=self.up(out_feat2)
        temp22 = torch.cat([out_feat33,up2], 1)  
        temp11 = torch.cat([out_feat22,up1], 1)  
        feat22=self.relu(self.cov1_temp2(temp22))
        feat11=self.relu(self.cov1_temp1(temp11))
        up1_temp = self.up_temp1(feat11, feat22)
        #------------------------------------------------------------



        final_s2 = self.final_1s2(up1) 
        final_vv = self.final_2vv(up1RH)
        final_temp=self.final_temp(up1_temp)

        up1_final = torch.cat([up1,up1RH,up1_temp], 1)
        up1_final= self.relu(self.cov1(up1_final) )
        up1_final= up1_final + self.relu(self.cov1_1(up1_final) )

        final = self.final(up1_final)
        
        return final,final_s2,final_vv,final_temp,ws2,ws1 




    def unfreeze_backbone(self):
        for param in self.owtDnet.parameters():
            param.requires_grad = True

