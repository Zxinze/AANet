import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .partialconv2d import PartialConv2d

"""To Do:
4. Deal with batchnorm freezing
5. Eventually do data augmentation on the masks and images (dilation, etc)
6. Run on a downsampled 128x128 version for a day to see how it goes
"""

class PartialLayer(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride, non_linearity="relu", bn=True, multi_channel=False):
        super(PartialLayer, self).__init__()

        self.conv = PartialConv2d(in_size, out_size, kernel_size, stride, return_mask=True, padding=(kernel_size - 1) // 2, multi_channel=multi_channel, bias=not bn)
        
        self.bn = nn.BatchNorm2d(out_size) if bn else None
        
        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        elif non_linearity == "leaky":
           self.non_linearity = nn.LeakyReLU(negative_slope=0.2)
        elif non_linearity == 'sigmoid':
            self.non_linearity = nn.Sigmoid()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
        elif non_linearity is None:
            self.non_linearity = None
        else:
            raise ValueError("unexpected value for non_linearity")

    def forward(self, x, mask_in=None, return_mask=True):
        x, mask = self.conv(x, mask_in=mask_in)
        
        if self.bn:
            x = self.bn(x)

        if self.non_linearity:
            x = self.non_linearity(x)

        if return_mask:
            return x, mask
        else:
            return x


class Model(nn.Module):
    def __init__(self, freeze_bn=False):
        super(Model, self).__init__()

        self.freeze_bn = freeze_bn # freeze bn layers for fine tuning

        self.conv1 = PartialLayer(3, 64, 7, 2) # encoder for UNET,  use relu for encoder
        self.conv2 = PartialLayer(64, 128, 5, 2)
        self.conv3 = PartialLayer(128, 256, 5, 2)
        self.conv4 = PartialLayer(256, 512, 3, 2)
        self.conv5 = PartialLayer(512, 512, 3, 2)
        self.conv6 = PartialLayer(512, 512, 3, 2)
        self.conv7 = PartialLayer(512, 512, 3, 2)
        self.conv8 = PartialLayer(512, 512, 3, 2)

        self.conv9 = PartialLayer(2 * 512, 512, 3, 1, non_linearity="leaky", multi_channel=True) # decoder for UNET
        self.conv10 = PartialLayer(2 * 512, 512, 3, 1, non_linearity="leaky", multi_channel=True)
        self.conv11 = PartialLayer(2 * 512, 512, 3, 1, non_linearity="leaky", multi_channel=True)
        self.conv12 = PartialLayer(2 * 512, 512, 3, 1, non_linearity="leaky", multi_channel=True)
        self.conv13 = PartialLayer(512 + 256, 256, 3, 1, non_linearity="leaky", multi_channel=True)
        self.conv14 = PartialLayer(256 + 128, 128, 3, 1, non_linearity="leaky", multi_channel=True)
        self.conv15 = PartialLayer(128 + 64, 64, 3, 1, non_linearity="leaky", multi_channel=True)
        self.conv16 = PartialLayer(64 + 3, 3, 3, 1, non_linearity="tanh", bn=False, multi_channel=True)

    def forward(self, x, mask):
        x1, mask1 = self.conv1(x, mask_in=mask)
        x2, mask2 = self.conv2(x1, mask_in=mask1)
        x3, mask3 = self.conv3(x2, mask_in=mask2)
        x4, mask4 = self.conv4(x3, mask_in=mask3)
        x5, mask5 = self.conv5(x4, mask_in=mask4)
        x6, mask6 = self.conv6(x5, mask_in=mask5)
        x7, mask7 = self.conv7(x6, mask_in=mask6)
        x8, mask8 = self.conv8(x7, mask_in=mask7)
        
        x9, mask9 = self.conv9(self.concat(x8, x7), mask_in=self.repeat(self.concat(mask8, mask7), 512, 512))
        x10, mask10 = self.conv10(self.concat(x9, x6), mask_in=self.repeat(self.concat(mask9, mask6), 512, 512))
        x11, mask11 = self.conv11(self.concat(x10, x5), mask_in=self.repeat(self.concat(mask10, mask5), 512, 512))
        x12, mask12 = self.conv12(self.concat(x11, x4), mask_in=self.repeat(self.concat(mask11, mask4), 512, 512))
        x13, mask13 = self.conv13(self.concat(x12, x3), mask_in=self.repeat(self.concat(mask12, mask3), 512, 256))
        x14, mask14 = self.conv14(self.concat(x13, x2), mask_in=self.repeat(self.concat(mask13, mask2), 256, 128))
        x15, mask15 = self.conv15(self.concat(x14, x1), mask_in=self.repeat(self.concat(mask14, mask1), 128, 64))
        out, mask16 = self.conv16(self.concat(x15, x), mask_in=self.repeat(self.concat(mask15, mask), 64, 3))

        return out

    def repeat(self, mask, size1, size2):
        return torch.cat([mask[:,0].unsqueeze(1).repeat(1, size1, 1, 1), mask[:,1].unsqueeze(1).repeat(1, size2, 1, 1)], dim=1)

    def concat(self, input, prev):
        return torch.cat([F.interpolate(input, scale_factor=2), prev], dim=1)


    def train(self, mode=True):
        super(Model, self).train(mode)

        if self.freeze_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and name[0:6] in ["conv" + str(n) + "." for n in range(1, 9)]:
                    # print("freezing layer {}".format(name))
                    module.eval()

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()

        vgg16 = models.vgg16(pretrained=True)

        self.layer1 = nn.Sequential(*vgg16.features[:5])
        self.layer2 = nn.Sequential(*vgg16.features[5:10])
        self.layer3 = nn.Sequential(*vgg16.features[10:17])

        for layer in [self.layer1, self.layer2, self.layer3]:
            for param in layer.parameters():
                param.requires_grad=False
        
    def forward(self, image):
        output = []

        for layer in [self.layer1, self.layer2, self.layer3]:
            image = layer(image)
            output.append(image)

        return output


if __name__ == "__main__":
    mask = torch.randint(0, 2, (10, 1, 512, 512)).float().cuda()
    data = torch.randn(10, 3, 512, 512).cuda()
    model = Model().cuda()

    out = model(data, mask)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    loss = F.l1_loss(out, data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert (torch.sum(torch.isnan(model.conv1.conv.weight.grad)).item() == 0)