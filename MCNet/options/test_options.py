from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        ###parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        ###parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc(choose folder in dataset)')
        # Dropout and Batchnorm has different behavioir during training and test.
        ###parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--eval', action='store_false', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=float("inf"), help='default=float("inf"),how many test images to run')

        parser.add_argument('--data_type', type=str, default='short',help='chooses data type in dataset.[short|long|]')


        # To avoid cropping, the load_size should be the same as crop_size
        ###parser.set_defaults(load_size=parser.get_default('crop_size'))

        # rewrite devalue values
        parser.set_defaults(num_threads = 0 ) # test code only supports num_threads = 1
        parser.set_defaults(batch_size = 1) # test code only supports batch_size = 1
        parser.set_defaults(serial_batches = True) #disable data shuffling; comment this line if results on randomly chosen images are needed.
        #parser.set_defaults(flip = False)# no flip; comment this line if results on flipped images are needed.
        ###parser.set_defaults(display_id = -1)# no visdom display; the test code saves the results to a HTML file.

        self.isTrain = False
        return parser
