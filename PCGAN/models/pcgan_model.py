import torch
from .base_model import BaseModel
from . import networks
from loss import InpaintingLoss
from loss import  Get_gradient_nopadding as GetGradient
import ntpath  ###

class PCGANModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.
    """


    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.result_names = ['real_A', 'mask_M', 'fake_B', 'final_B','real_B', 'refer_R','fake_B_gradient', 'real_B_gradient'] ###
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        #self.model_names = ['PCU']
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        #self.loss_names = ['PixWise']
        #self.netPCU = networks.PCUnet(opt.input_nc, opt.output_nc, self.gpu_ids)
        self.gradient = GetGradient()

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.generator(opt.input_nc, opt.output_nc, opt.enc_cc, opt.dec_cc, self.gpu_ids)


        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.discriminator(opt.input_nc + opt.output_nc + 1, opt.ndf,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = InpaintingLoss(networks.VGG16FeatureExtractor()).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


        # if self.isTrain:
        #     # define loss functions
        #     self.criterionLoss = InpaintingLoss(networks.VGG16FeatureExtractor()).to(self.device)
        #     # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        #     self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netPCU.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        #     self.optimizers = [self.optimizer]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.mask_M = input['M'].to(self.device)
        self.refer_R = input['R'].to(self.device)
        self.image_paths = input['paths']
        self.image_names = ntpath.basename(input['paths'][0])

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, _ = self.netG(self.real_A, self.mask_M)  #
        self.final_B = self.real_A + self.fake_B * (1.0 - self.mask_M)
        self.fake_B_gradient = self.gradient(self.fake_B)
        self.real_B_gradient = self.gradient(self.real_B)
        #print(self.fake_B_gradient.shape)
    # def backward(self):
    #     """Calculate losses, gradients, and update network weights; called in every training iteration"""
    #     # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
    #     # calculate loss given the input and intermediate results
    #     self.loss_PixWise = self.criterionLoss(self.real_A, self.mask_M, self.fake_B, self.real_B)
    #     self.loss_PixWise.backward()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_ABBg = torch.cat((self.real_A, self.fake_B, self.fake_B_gradient), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_ABBg.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_ABBg = torch.cat((self.real_A, self.real_B, self.real_B_gradient), 1)
        pred_real = self.netD(real_ABBg)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * 0.1
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_ABBg = torch.cat((self.real_A, self.fake_B, self.fake_B_gradient), 1)
        pred_fake = self.netD(fake_ABBg)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 0.1
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.real_A, self.mask_M, self.fake_B, self.real_B, self.fake_B_gradient, self.real_B_gradient)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights


    # def optimize_parameters(self):
    #     """Update network weights; it will be called in every training iteration."""
    #     self.forward()  # first call forward to calculate intermediate results
    #     self.optimizer.zero_grad()  # clear network PCU's existing gradients
    #     self.backward()  # calculate gradients for network PCU
    #     self.optimizer.step()  # update gradients for network PCU

