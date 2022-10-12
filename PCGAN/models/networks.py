import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .partialconv2d import PartialConv2d

from .cc_attention import CrissCrossAttention
###############################################################################
# Helper Functions
###############################################################################

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module





def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones= opt.milestones, gamma=opt.step_gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler




def generator(input_nc, output_nc, enc_cc, dec_cc, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images

    The generator has been initialized  in Class GeneratorPConvUNet
    """

    net = PConvUNet(input_channels=input_nc, output_channels=output_nc, enc_cc=enc_cc, dec_cc=dec_cc)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        #print(net)

    return net


# def PCUnet(input_nc, output_nc, gpu_ids=[]):
#     """Create a generator
#
#     Parameters:
#         input_nc (int) -- the number of channels in input images
#         output_nc (int) -- the number of channels in output images
#
#     The generator has been initialized  in Class GeneratorPConvUNet
#     """
#
#     net = PConvUNet(input_channels=input_nc, output_channels=output_nc)
#
#     if len(gpu_ids) > 0:
#         assert(torch.cuda.is_available())
#         net.to(gpu_ids[0])
#         net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
#         #print(net)
#
#     return net


def discriminator(input_nc, ndf, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)


    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    norm_layer = get_norm_layer(norm_type=norm)
    net = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer)
    #print(net)

    return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


########################################################################


class PartialLayerSN(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride, non_linearity="relu", bn=True, multi_channel=False):
        super(PartialLayerSN, self).__init__()

        self.conv = spectral_norm(PartialConv2d(in_size, out_size, kernel_size, stride, return_mask=True,
                                  padding=(kernel_size - 1) // 2, multi_channel=multi_channel, bias=not bn))

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


class PConvUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, enc_cc='', dec_cc='', freeze_bn=False):
        super(PConvUNet, self).__init__()

        self.freeze_bn = freeze_bn  # freeze bn layers for fine tuning
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.enc_cc = [int(i) for i in enc_cc.split(',')] if len(enc_cc) else []
        self.dec_cc = [int(i) for i in dec_cc.split(',')] if len(dec_cc) else []

        self.conv1 = PartialLayerSN(self.input_channels, 64, 7, 2, bn=False)  # encoder for UNET,  use relu for encoder
        self.conv2 = PartialLayerSN(64, 128, 5, 2)
        self.conv3 = PartialLayerSN(128, 256, 5, 2)
        self.conv4 = PartialLayerSN(256, 512, 3, 2)
        self.conv5 = PartialLayerSN(512, 512, 3, 2)
        self.conv6 = PartialLayerSN(512, 512, 3, 2)
        self.conv7 = PartialLayerSN(512, 512, 3, 2)

        self.conv8 = PartialLayerSN(2 * 512, 512, 3, 1, non_linearity="leaky", multi_channel=True) # decoder for UNET
        self.conv9 = PartialLayerSN(2 * 512, 512, 3, 1, non_linearity="leaky", multi_channel=True)
        self.conv10 = PartialLayerSN(2 * 512, 512, 3, 1, non_linearity="leaky", multi_channel=True)
        self.conv11 = PartialLayerSN(512 + 256, 256, 3, 1, non_linearity="leaky", multi_channel=True)
        self.conv12 = PartialLayerSN(256 + 128, 128, 3, 1, non_linearity="leaky", multi_channel=True)
        self.conv13 = PartialLayerSN(128 + 64, 64, 3, 1, non_linearity="leaky", multi_channel=True)
        self.conv14 = PartialLayerSN(64 + self.input_channels, self.output_channels, 3, 1, non_linearity="tanh",
                                     bn=False, multi_channel=True)

        for i in self.enc_cc:
            name = 'enc_cc_{:d}'.format(i)
            setattr(self, name, CrissCrossAttention(getattr(self,'conv{:d}'.format(i)).conv.out_channels))

        for i in self.dec_cc:
            name = 'dec_cc_{:d}'.format(i+1)
            setattr(self, name, CrissCrossAttention(getattr(self,'conv{:d}'.format(15-(i+1))).conv.out_channels))

    def forward(self, x, mask):

        x1, mask1 = self.conv1(x, mask_in=mask)
        x1        = self.enc_cc_1(x1) if 1 in self.enc_cc else x1
        x2, mask2 = self.conv2(x1, mask_in=mask1)
        x2        = self.enc_cc_2(x2) if 2 in self.enc_cc else x2
        x3, mask3 = self.conv3(x2, mask_in=mask2)
        x3        = self.enc_cc_3(x3) if 3 in self.enc_cc else x3
        x4, mask4 = self.conv4(x3, mask_in=mask3)
        x4        = self.enc_cc_4(x4) if 4 in self.enc_cc else x4
        x5, mask5 = self.conv5(x4, mask_in=mask4)
        x5        = self.enc_cc_5(x5) if 5 in self.enc_cc else x5
        x6, mask6 = self.conv6(x5, mask_in=mask5)
        x6        = self.enc_cc_6(x6) if 6 in self.enc_cc else x6
        x7, mask7 = self.conv7(x6, mask_in=mask6)
        x7        = self.enc_cc_7(x7) if 7 in self.enc_cc else x7

        x8, mask8 = self.conv8(self.concat(x7, x6), mask_in=self.repeat(self.concat(mask7, mask6), 512, 512))
        x8        = self.dec_cc_7(x8) if 6 in self.dec_cc else x8
        x9, mask9 = self.conv9(self.concat(x8, x5), mask_in=self.repeat(self.concat(mask8, mask5), 512, 512))
        x9        = self.dec_cc_6(x9) if 5 in self.dec_cc else x9
        x10, mask10 = self.conv10(self.concat(x9, x4), mask_in=self.repeat(self.concat(mask9, mask4), 512, 512))
        x10        = self.dec_cc_5(x10) if 4 in self.dec_cc else x10
        x11, mask11 = self.conv11(self.concat(x10, x3), mask_in=self.repeat(self.concat(mask10, mask3), 512, 256))
        x11        = self.dec_cc_4(x11) if 3 in self.dec_cc else x11
        x12, mask12 = self.conv12(self.concat(x11, x2), mask_in=self.repeat(self.concat(mask11, mask2), 256, 128))
        x12        = self.dec_cc_3(x12) if 2 in self.dec_cc else x12
        x13, mask13 = self.conv13(self.concat(x12, x1), mask_in=self.repeat(self.concat(mask12, mask1), 128, 64))
        x13        = self.dec_cc_2(x13) if 1 in self.dec_cc else x13
        out, mask14 = self.conv14(self.concat(x13, x), mask_in=self.repeat(self.concat(mask13, mask), 64, self.output_channels))

        return out, mask14

    def repeat(self, mask, size1, size2):
        return torch.cat(
            [mask[:, 0].unsqueeze(1).repeat(1, size1, 1, 1), mask[:, 1].unsqueeze(1).repeat(1, size2, 1, 1)], dim=1)

    def concat(self, input, prev):
        return torch.cat([F.interpolate(input, scale_factor=2), prev], dim=1)

    def setfreeze_bn(self, boolvalue):
        self.freeze_bn = boolvalue

    def train(self, mode=True):
        super(PConvUNet, self).train(mode)
        print('Override the default train() to freeze the BN parameters')
        print(self.freeze_bn)

        if self.freeze_bn:
            for name, module in self.named_modules():
                #if isinstance(module, nn.BatchNorm2d) and name[0:6] in ["conv" + str(n) + "." for n in range(1, 9)]:
                if isinstance(module, nn.BatchNorm2d) and name[0:6] in ["conv" + str(n) + "." for n in range(1, 8)]:
                    #print("freezing layer {}".format(name))
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
                param.requires_grad = False

    def forward(self, image):
        output = []

        for layer in [self.layer1, self.layer2, self.layer3]:
            image = layer(image)
            output.append(image)

        return output



def get_norm_layer(norm_type='batch'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True))]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)




