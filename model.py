
import torch
import torch.nn as nn
from backbone.pvtv2 import pvt_v2_b2
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Channel_compress(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Channel_compress, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.convf = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(2*channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(3*channels, channels, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(4*channels, channels, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1))
    def forward(self, x3, x4, x5, x6):
        x3 = self.upsample(x3)
        x4 = self.conv4(torch.cat((x3, x4), dim=1))
        x3 = self.upsample(x3)
        x4 = self.upsample(x4)
        x5 = self.conv5(torch.cat((x3, x4, x5), dim=1))
        x5 = self.upsample(x5)
        x4 = self.upsample(x4)
        x3 = self.upsample(x3)
        x6 = self.conv6(torch.cat((x3, x4, x5, x6), dim=1))
        x6=self.convf(x6)
        return x6

class Fusion(nn.Module):
    def __init__(self,k):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(k,int(k/2),3,1,1)
        self.bn1=nn.BatchNorm2d(int(k/2))
        self.conv2 = nn.Conv2d(k, int(k / 2), 3, 1, 1)
        self.bn2= nn.BatchNorm2d(int(k / 2))
        self.weight=nn.Conv2d(k,2,1,1)


    def forward(self,rgb,depth):
        input_rgb=rgb
        input_depth=depth
        rgb=self.relu(self.bn1(self.conv1(rgb)))
        depth=self.relu(self.bn2(self.conv2(depth)))
        fusion=torch.cat((rgb,depth),dim=1)
        fusion=self.weight(fusion)
        fusion_weight=F.softmax(fusion,dim=1)
        out=fusion_weight[:,0:1,:,:]*input_rgb+fusion_weight[:,1:,:,:]*input_depth
        return out

class USODModel(nn.Module):
    def __init__(self):
        super(USODModel, self).__init__()
        #rgb
        self.rgb_encoder = pvt_v2_b2()
        self.rgb3 = Channel_compress(512, 64)
        self.rgb2 = Channel_compress(320, 64)
        self.rgb1 = Channel_compress(128, 64)
        self.rgb0 = Channel_compress(64, 64)

        #depth
        self.depth_encoder = pvt_v2_b2()
        self.depth3 = Channel_compress(512, 64)
        self.depth2 = Channel_compress(320, 64)
        self.depth1 = Channel_compress(128, 64)
        self.depth0 = Channel_compress(64, 64)

        self.fusion0 = Fusion(64)
        self.fusion1 = Fusion(64)
        self.fusion2 = Fusion(64)
        self.fusion3 = Fusion(64)

        self.decoder=Decoder(64)
        self.decoder_rgb=Decoder(64)
        self.decoder_depth=Decoder(64)
    def forward_step(self, r,d):
        #rgb
        r = self.rgb_encoder(r)
        r[0] = self.rgb0(r[0])  # [ba, 32, 64, 64]
        r[1] = self.rgb1(r[1])  # [ba, 32, 32, 32]
        r[2] = self.rgb2(r[2])  # [ba, 32, 16, 16]
        r[3] = self.rgb3(r[3])  # [ba, 32, 8, 8]

        #depth
        d = self.depth_encoder(d)
        d[0] = self.depth0(d[0])  # [ba, 32, 64, 64]
        d[1] = self.depth1(d[1])  # [ba, 32, 32, 32]
        d[2] = self.depth2(d[2])  # [ba, 32, 16, 16]
        d[3] = self.depth3(d[3])  # [ba, 32, 8, 8]

        f0=self.fusion0(r[0],d[0])
        f1 = self.fusion1(r[1], d[1])
        f2 = self.fusion2(r[2], d[2])
        f3 = self.fusion0(r[3], d[3])

        # rgb energy
        rgb_energy0 = (-0.1) * -torch.log(torch.exp(r[0]) + torch.exp(1 - r[0]))
        rgb_energy1 = (-0.1) * -torch.log(torch.exp(r[1]) + torch.exp(1 - r[1]))
        rgb_energy2 = (-0.1) * -torch.log(torch.exp(r[2]) + torch.exp(1 - r[2]))
        rgb_energy3 = (-0.1) * -torch.log(torch.exp(r[3]) + torch.exp(1 - r[3]))

        # depth_energy
        depth_energy0 = (-0.1) * -torch.log(torch.exp(d[0]) + torch.exp(1 - d[0]))
        depth_energy1 = (-0.1) * -torch.log(torch.exp(d[1]) + torch.exp(1 - d[1]))
        depth_energy2 = (-0.1) * -torch.log(torch.exp(d[2]) + torch.exp(1 - d[2]))
        depth_energy3 = (-0.1) * -torch.log(torch.exp(d[3]) + torch.exp(1 - d[3]))

        f0 = rgb_energy0 * r[0] + depth_energy0 * d[0]
        f1 = rgb_energy1 * r[1] + depth_energy1 * d[1]
        f2 = rgb_energy2 * r[2] + depth_energy2 * d[2]
        f3 = rgb_energy3 * r[3] + depth_energy3 * d[3]

        rgb_depth_out = self.decoder(f3, f2, f1, f0)  # [2, 384, 64, 64]
        rgb_out = self.decoder_rgb(r[3],r[2],r[1],r[0])
        depth_out=self.decoder_depth(d[3],d[2],d[1],d[0])
        rgb_energy=torch.mean(rgb_energy3,dim=1)
        depth_energy=torch.mean(depth_energy3,dim=1)

        rgb_depth_out=F.interpolate(rgb_depth_out,size=(256,256))
        rgb_out=F.interpolate(rgb_out,size=(256,256))
        depth_out=F.interpolate(depth_out,size=(256,256))

        return (rgb_depth_out,rgb_out,depth_out,rgb_energy,depth_energy)


