""" 
Add the configurations by modules

@author: Zhaoyang Lv
@date: March 2019
"""
import argparse
import LEM_SFM.config as config



def get_model_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--device', help='device,在train.py中定义')
    parser.add_argument('--max_iter_per_pyr',
        default=3, type=int,
        help='The maximum number of iterations at each pyramids.\n')
    
    return parser.parse_args()


def add_tracking_config(parser):
    parser.add_argument('--network',
        default='DeepIC', type=str,
        choices=('DeepIC', 'GaussNewton'),
        help='Choose a network to run. \n \
        The DeepIC is the proposed Deeper Inverse Compositional method. \n\
        The GuassNewton is the baseline for Inverse Compositional method which does not include \
        any learnable parameters\n')
    parser.add_argument('--mestimator',
        default='None', type=str,
        choices=('None', 'MultiScale2w'),
        help='Choose a weighting function for the Trust Region method.\n\
            The MultiScale2w is the proposed (B) convolutional M-estimator. \n')
    parser.add_argument('--solver',
        default='Direct-ResVol', type=str,
        choices=('Direct-Nodamping', 'Direct-ResVol'),
        help='Choose the solver function for the Trust Region method. \n\
            Direct-Nodamping is the Gauss-Newton algorithm, which does not use damping. \n\
            Direct-ResVol is the proposed (C) Trust-Region Network. \n\
            (default: Direct-ResVol) ')
    parser.add_argument('--encoder_name',
        default='ConvRGBD2',
        choices=('ConvRGBD2', 'RGB'),
        help='The encoder architectures. \
            ConvRGBD2 takes the two-view features as input. \n\
            RGB is using the raw RGB images as input (converted to intensity afterwards).\n\
            (default: ConvRGBD2)')
    parser.add_argument('--max_iter_per_pyr',
        default=3, type=int,
        help='The maximum number of iterations at each pyramids.\n')
    parser.add_argument('--no_weight_sharing',
        action='store_true',
        help='If this flag is on, we disable sharing the weights across different backbone network when extracing \
         features. In default, we share the weights for all network in each pyramid level.\n')
    parser.add_argument('--tr_samples', default=10, type=int,
        help='Set the number of trust-region samples. (default: 10)\n')

def add_basics_config(parser):
    """ the basic setting
    (supposed to be shared through train and inference)
    """
    # parser.add_argument('--cpu_workers', type=int, default=0,
    #     help="Number of cpu threads for data loader.\n")
    # parser.add_argument('--dataset', type=str, default='TUM_RGBD',
    #     choices=('TUM_RGBD', 'MovingObjects3D'),
    #     help='Choose a dataset to train/val/evaluate.\n')
    parser.add_argument('--time', dest='time', action='store_true',
        help='Count the execution time of each step.\n' )
    parser.add_argument('--checkpoint', 
        type=str, help='the path to the pre-trained checkpoint.')
    parser.add_argument('--model_type', type=str,  default='deepICN',choices=('deepICN','None'))

def add_test_basics_config(parser):
    # parser.add_argument('--batch_per_gpu', default=8, type=int,
    #     help='Specify the batch size during test. The default is 8.\n')
    # parser.add_argument('--checkpoint', default='', type=str,
    #     help='Choose a checkpoint model to test.\n')
    # parser.add_argument('--keyframes',
    #     default='1,2,4,8', type=str,
    #     help='Choose the number of keyframes to train the algorithm.\n')
    parser.add_argument('--verbose', action='store_true',
        help='Print/save all the intermediate representations')

    # parser.add_argument('--trajectory', type=str, 
    #     default = '',
    #     help = 'Specify a trajectory to run.\n')

def add_train_basics_config(parser):
    """ add the basics about the training """
    #要不要加载checkpoint
    # parser.add_argument('--checkpoint', default='', type=str,
    #     help='Choose a pretrained checkpoint model to start with. \n')
    # parser.add_argument('--batch_per_gpu', default=64, type=int,
    #     help='Specify the batch size during training.\n')
    # parser.add_argument('--epochs',
    #     default=30, type=int,
    #     help='The total number of total epochs to run. Default is 30.\n' )
    parser.add_argument('--resume_training',
        dest='resume_training', action='store_true',
        help='Resume the training using the loaded checkpoint. If not, restart the training. \n\
            You will need to use the --checkpoint config to load the pretrained checkpoint' )
    parser.add_argument('--pretrained_model', default='', type=str,
        help='Initialize the model weights with pretrained model.\n')
    parser.add_argument('--no_val',
        default=False,
        action='store_true',
        help='Use no validatation set for training.\n')
    # parser.add_argument('--keyframes',
    #     default='1,2,4', type=str,
    #     help='Choose the number of keyframes to train the algorithm')
    parser.add_argument('--verbose', action='store_true',
        help='Print/save all the intermediate representations.\n')

def add_train_log_config(parser):
    """ checkpoint and log options """
    parser.add_argument('--checkpoint_folder', default='', type=str,
        help='The folder name (postfix) to save the checkpoint.')
    parser.add_argument('--snapshot', default=1, type=int,
        help='Number of interations to save a snapshot')
    #保存频率
    parser.add_argument('--save_checkpoint_freq',
        default=1, type=int,
        help='save the checkpoint for every N epochs')
    parser.add_argument('--prefix', default='', type=str,
        help='the prefix string added to the log files')
    parser.add_argument('-p', '--print_freq',
        default=10, type=int,
        help='print frequency (default: 10)')

def add_train_optim_config(parser):
    """ add training optimization options """
    parser.add_argument('--opt',
        type=str, default='adam', choices=('sgd','adam'),
        help='choice of optimizer (default: adam) \n')
    parser.add_argument('--lr',
        default=5*1e-4, type=float,
        help='initial learning rate. \n')
    parser.add_argument('--lr_decay_ratio',
        default=0.5, type=float,
        help='lr decay ratio (default:0.5)')
    parser.add_argument('--lr_decay_epochs',
        default=[5, 10, 20], type=int, nargs='+',
        help='lr decay epochs')
    parser.add_argument('--lr_min', default=1e-6, type=float,
        help='minimum learning rate')
    parser.add_argument('--lr_restart', default=10, type=int,
        help='restart learning after N epochs')

def add_train_loss_config(parser):
    """ add training configuration for the loss function """
    parser.add_argument('--regression_loss_type',
        default='SmoothL1', type=str, choices=('L1', 'SmoothL1'),
        help='Loss function for flow regression (default: SmoothL1 loss)')
    



def get_depth_args():
    parser = argparse.ArgumentParser(description='get_depth_args')
    # parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--checkpoint', '-f', type=str, default='relative_depth/checkpoint/checkpoint_epoch193.pth', help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--val', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_false', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_false', default=True, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    parser.add_argument('--model_type', type=str,  default='relative_depth',choices=('DepthAnything','relative_depth','MonSter','None'),)
    parser.add_argument('--DepthAnything_encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])

    return parser.parse_args()


def get_pose_args_refer():
    parser = argparse.ArgumentParser(description='Run the network inference example.')

    parser.add_argument('--checkpoint', 
        type=str, help='the path to the pre-trained checkpoint.')

    # parser.add_argument('--color_dir', default='data/data_examples/TUM/color',
    #     help='the directory of color images')
    # parser.add_argument('--depth_dir', default='data/data_examples/TUM/depth', 
    #     help='the directory of depth images')

    parser.add_argument('--intrinsic', default='525.0,525.0,319.5,239.5', 
        help='Simple pin-hole camera intrinsics, input in the format (fx, fy, cx, cy)')
    
    config.add_tracking_config(parser)

    return parser.parse_args()

def get_pose_args_train():
    parser = argparse.ArgumentParser(description='Training the network')
    config.add_basics_config(parser)

    config.add_tracking_config(parser)


    return parser.parse_args()


def get_pose_args_evaluate():
    parser = argparse.ArgumentParser(description="Evaluate the network")
    config.add_basics_config(parser)
    config.add_test_basics_config(parser)
    config.add_tracking_config(parser)

    return parser.parse_args()

def get_args():
    parser = argparse.ArgumentParser(description='get_args')
    parser.add_argument('--device', default='',help='')
    parser.add_argument('--dataroot',help='')
    
    parser.add_argument('--batch_per_gpu', default=64, type=int,
        help='Specify the batch size during test. The default is 8.\n')
    parser.add_argument('--keyframes',
        default='1,2,4,8', type=str,
        help='Choose the number of keyframes to train the algorithm.\n')
    
    parser.add_argument('--dataset', type=str, default='TUM_RGBD',
        choices=('TUM_RGBD', 'MovingObjects3D'),
        help='Choose a dataset to train/val/evaluate.\n')
    parser.add_argument('--cpu_workers', type=int, default=32,
        help="Number of cpu threads for data loader.\n")
    parser.add_argument('--trajectory', default='')

    parser.add_argument('--epochs',
        default=35, type=int,
        help='The total number of total epochs to run. Default is 30.\n' )
    
    parser.add_argument('--start_epoch',
        default=9, type=int,
        help='\n' )
    
    parser.add_argument('--eval_set', default='test',
        choices=('test', 'val'))
    
    parser.add_argument('--checkpoint_Union', default='', 
        type=str, help='the path to the pre-trained checkpoint.')

    parser.add_argument('--time', dest='time', action='store_true',
        help='Count the execution time of each step.\n' )
    config.add_train_basics_config(parser)
    config.add_train_optim_config(parser)
    config.add_train_log_config(parser)
    config.add_train_loss_config(parser)

    parser.add_argument('--noise_gyro' , default=0.1)
    parser.add_argument('--noise_accel' , default=0.5)
    parser.add_argument('--speed' , default=1)
    parser.add_argument('--dt' , default=0.1)

    parser.add_argument('--batch_size', default=64, 
                        type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--num_workers', default=8, type=int)

    return parser.parse_args()


def get_monster_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--restore_ckpt', help="restore checkpoint", default="/home/DeepCompose2/code/MonSter/checkpoints/mix_all.pth")
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)

    # parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow', choices=["eth3d", "kitti", "sceneflow", "vkitti", "driving"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    return parser.parse_args()




def get_Deeplabv3_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options

    parser.add_argument("--model_type", type=str, default='Deeplabv3')
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--save_val_results_path",type = str,  default='./results/',
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser.parse_args()