"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import os
import sys
import datetime


from util.util import save_current_images

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    numb_iters_epoch = dataset_size // opt.batch_size
    print('The number of training images = %d, iters = %d' % (dataset_size, numb_iters_epoch))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    total_iters = 0                # the total number of training iterations
    save_image_freq_iter= numb_iters_epoch//1 ### the frequency of saving image in a epoch
    prev_time = time.time()

    start_epoch = opt.epoch_count
    if opt.continue_train ==True:
       # update learning_rate
       temp = model.metric
       model.metric = opt.epoch-1  # class _LRScheduler  .step() function, self.epoch starts from -1
       model.update_learning_rate()
       model.metric = temp
       # continue trainning
       start_epoch = opt.epoch + 1  ###  next epoch

    model.netG.module.setfreeze_bn(boolvalue = False)
                           ###freeze the Batch Normalization parameters in the encoder part of generator.
                           ### start from last stage of --milestones
    ##for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(start_epoch, opt.epoch_total + 1):

        print(opt.milestones[-1], model.netG.module.freeze_bn)
        if opt.fine_tune==True and epoch > opt.milestones[-1]:
            #print(type(model.netPCU))
            model.netG.module.setfreeze_bn(boolvalue = True)

        model.netG.train()

        #model.update_learning_rate()  # update learning rates at the start of every epoch.

        epoch_iter = 0    # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch

            total_iters = total_iters +1
            epoch_iter = epoch_iter + 1

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if epoch_iter % save_image_freq_iter ==0:
                save_result_dir = os.path.join(opt.checkpoints_dir,'%s_image_result' % (opt.phase))
                if os.path.exists(save_result_dir) is False:
                    os.makedirs(save_result_dir)
                img_path = os.path.join(save_result_dir, '%s_%s_%s.png' % (epoch, epoch_iter, opt.phase))
                results = model.get_current_results()
                save_current_images(results, img_path)

            # Determine approximate time left
            batches_done = epoch * numb_iters_epoch + i -1
            batches_left = opt.epoch_total * numb_iters_epoch - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print log
            losses = model.get_current_losses()
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [G_GAN_loss: %f, G_L1_loss: %f] [D_real_loss: %f, D_fake_loss: %f] ETA: %s" %
                (epoch, opt.epoch_total,
                 i + 1, numb_iters_epoch,
                 losses['G_GAN'], losses['G_L1'],
                 losses['D_real'], losses['D_fake'],
                 time_left))
            '''
            save_model_freq_iter = numb_iters_epoch // 2 + 1  ### the frequency of saving model in a epoch
            if epoch_iter % save_model_freq_iter == 0:   # cache our latest model every <save_latest_freq> iterations
                print('\n')
                print('saving the latest model (Epoch %d, Iters %d)' % (epoch, epoch_iter))
                save_suffix = '%d_%d' % (epoch, epoch_iter)
                model.save_networks(save_suffix)
            '''
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('\nsaving the model at the end of Epoch %d, Iters %d\n' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()  # update learning rates at the start of every epoch.


