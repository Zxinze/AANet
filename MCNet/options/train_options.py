from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model and SetUp --epoch')
        parser.add_argument('--epoch', type=int, default=1, help='the starting epoch count, when continue_train')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch>+<save_latest_freq>, ...')
        parser.add_argument('--epoch_total', type=int, default=200, help='the number of epoch of training model  ')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc(choose folder in dataset)')
        parser.add_argument('--fine_tune', type=bool, default=True,  help='freeze the Batch Normalization parameters in the encoder part of generator.'
                                                                          'start from last stage of --milestones  ')

        # training parameters
        ### total epoch = niter + niter_decay
        ###parser.add_argument('--niter', type=int, default=5, help='# of iter at starting learning rate(epoch)' related to lr_policy ==linear)
        ###parser.add_argument('--niter_decay', type=int, default=5, help='# of iter to linearly decay learning rate to zero (epoch)' related to lr_policy ==linear)
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan',
                            help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='multistep', help='learning rate policy. [linear | step | multistep| plateau | cosine]')
        ###parser.add_argument('--lr_decay_iters', type=int, default=5, help='multiply by a gamma every lr_decay_iters iterations(related to choice of --lr_policy ==step)')
        parser.add_argument('--step_gamma', type=float, default=0.5, help='when --lr_policy is step or multistep, --step_gamma setup scheduler ')
        parser.add_argument('--milestones', type=list, default=[100,150], help='when --lr_policy is multistep, --milestones setup scheduler ')

        # dataset parameters
        parser.add_argument('--data_type', type=str, default='short', help='chooses  data type in dataset.[short]')
        self.isTrain = True
        return parser
