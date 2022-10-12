import torch
import torch.nn as nn
import torch.nn.functional as F

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

'''
def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss
'''

def total_variation_loss(image, mask):
    dilated = F.conv2d(1 - mask, torch.ones((1, 1, 3, 3)).cuda(), padding=1)
    dilated[dilated != 0] = 1

    eroded = F.conv2d(mask, torch.ones((1, 1, 3, 3)).cuda(), padding=1)
    eroded[eroded != 0] = 1
    eroded = 1 - eroded

    mask = dilated - eroded # slightly more than the 1-pixel region in the paper, but okay

    # adapted from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/loss.py
    loss = (torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]) * mask[:,:,:,:-1]).mean() + \
        (torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]) * mask[:,:,:-1,:]).mean()

    return loss


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):

        x_v = F.conv2d(x, self.weight_v, padding = 1)
        x_h = F.conv2d(x, self.weight_h, padding = 1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x


class InpaintingLoss(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.lambda_dict = {'hole': 100.0, 'valid': 10.0,'tv': 0.1, 'prc': 0.05, 'style': 120.0,'gradient_hole':300.0, 'gradient_valid':10.0}

    def forward(self, input, mask, output, gt, output_gradient, gt_gradient):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('only gray an')

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp, mask)

        loss_dict['gradient_hole'] = self.l1((1 - mask)*output_gradient, (1 - mask)*gt_gradient)
        loss_dict['gradient_valid'] = self.l1(mask * output_gradient, mask * gt_gradient)

        # Pixel-wise loss
        loss_pixel = 0.0
        for key, coef in self.lambda_dict.items():
            value = coef * loss_dict[key]
            loss_pixel += value

        return loss_pixel
