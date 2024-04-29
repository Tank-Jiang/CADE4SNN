import argparse
from utils.tools import template
import yaml
import os
import torch
import logging
from contextlib import suppress
from utils.tools.utility import NativeScaler
from utils.data.cifar_loader import create_loader_cifar
from utils.data.imagenet_loader import create_loader
_logger = logging.getLogger(__name__)

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default=None, type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')   #'imagenet.yml'

parser = argparse.ArgumentParser(description='Finetune Training and Evaluating')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Dataset / Model parameters
parser.add_argument('--data', default='/data/dataset/imagenet',
                    help='path to dataset')
parser.add_argument('--exp_name', default='exp_debug', type=str, metavar='EXP',
                    help='path to exp_name folder (default: exp_debug, current dir)')
parser.add_argument('--model', default='resnet18', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--only_test', action='store_true', default=False,
                    help='if test only')
parser.add_argument('--num_classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--img_size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop_pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('-b', '--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--val_batch_size', type=int, default=96, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Augmentation & regularization parameters
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original or rand-m9-mstd0.5-inc1". (default: None)'),

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('-j', '--workers', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('-j_val', '--val_workers', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--output', default='/data/guodong/evo/output/', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--log_dir', default='/home/guodong/evo/log_out/demo.txt',
                    help='path to dataset')
# parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--local-rank", default=0, type=int)

parser.add_argument('--slice_len', type=int, default=0, metavar='slice_len',
                    help='slice_len (default: 0)')

# DE parameters
parser.add_argument('--de_slice_len', type=int, default=20000, metavar='N',
                    help='number of label classes (default: 20000)')
parser.add_argument('--de_batch_size', type=int, default=256, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--de_epochs', type=int, default=300, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--popsize', type=int, default=10, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--pop_init', default='/home/guodong/ImageNet',
                    help='path to dataset')
parser.add_argument('--shade_mem', type=int, default=5,
                    help='memory size of archive in shade')
parser.add_argument('--shade_lp', type=int, default=5,
                    help='lp in shade')
parser.add_argument('--trig_perc', type=float, default=0.2,
                    help='Override std deviation of of dataset')
parser.add_argument('--cr_init', type=float, default=0.8,
                    help='initial value of crossover rate cr')
parser.add_argument('--cr_std', type=float, default=0.05,
                    help='Override std deviation of of dataset')
parser.add_argument('--cr_clip', type=float, default=0.99,
                    help='Override std deviation of of dataset')
parser.add_argument('--f_init', type=float, default=0.6,
                    help='Override std deviation of of dataset')
parser.add_argument('--f_std', type=float, default=0.05,
                    help='Override std deviation of of dataset')
parser.add_argument('--f_clip', type=float, default=0.99,
                    help='Override std deviation of of dataset')
parser.add_argument('--freq_init', type=float, default=0.5,
                    help='Override std deviation of of dataset')
parser.add_argument('--freq_std', type=float, default=0.1,
                    help='Override std deviation of of dataset')
parser.add_argument('--test_ood', action='store_true', default=False,
                    help='process test on ood')
parser.add_argument('--ood_path', default='/home/guodong/runhua/ood_dataset',
                    help='path to ood dataset')
# parser.add_argument('--skip-init-validate', action='store_true', default=False,
#                     help='skip the initial pop validation')
# parser.add_argument('--skip-pop-validate', action='store_true', default=False,
#                     help='skip the pop validation')
# parser.add_argument('--replace-pop-after-epoch', action='store_true', default=False,
#                     help='replace half pop to initial pop after an epoch')
# parser.add_argument('--noise-parent', action='store_true', default=False,
#                     help='using noise to generate parents')
# parser.add_argument('--restrict-para', action='store_true', default=False,
#                     help='restrict the trainable parameters in DE')
# parser.add_argument('--validate-every-iter', action='store_true', default=False,
#                     help='validate the result for every iteration, otherwise every batch')
# parser.add_argument('--resume-parents', action='store_true', default=False,
#                     help='resume parent according to the parent_pop_dir')
# parser.add_argument('--validate_interval', type=int, default=1, metavar='N',
#                     help='no of epoch between validation')

def _parse_args(config_parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

args, args_text = _parse_args(config_parser)
template.set_template(args)

# if args.epochs == 0: args.epochs = 1e8
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

if getattr(torch.cuda.amp, 'autocast') is None:
    _logger.warning('amp have no autocast!')
args.distributed = False

if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
    print('distributed:', args.distributed)
    if args.distributed and args.num_gpu > 1:
         _logger.warning(
             'Using more than one GPU per process in distributed mode is not allowed. Setting num_gpu to 1.')
         args.num_gpu = 1

args.device = 'cuda:0'
args.world_size = 1
args.rank = 0  # global rank
if args.distributed:
   args.num_gpu = 1
   args.device = 'cuda:%d' % args.local_rank
   torch.cuda.set_device(args.local_rank)
   torch.distributed.init_process_group(backend='nccl', init_method='env://')
   args.world_size = torch.distributed.get_world_size()
   args.rank = torch.distributed.get_rank()
assert args.rank >= 0

if args.distributed:
    _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                 % (args.rank, args.world_size))
else:
    _logger.info('Training with a single process on %d GPUs.' % args.num_gpu)

amp_autocast = suppress  # do nothing
loss_scaler = None
if args.amp == True:
    amp_autocast = torch.cuda.amp.autocast
    loss_scaler = NativeScaler()
    if args.local_rank == 0:
        _logger.info('Using native Torch AMP. Training in mixed precision.')
else:
    if args.local_rank == 0:
        _logger.info('AMP not enabled. Training in float32.')

def obtain_loader(args):
    if args.num_classes == 1000:
        loader_train = create_loader(data=args.data, name='train98p', batch_size=args.batch_size,
                                    num_workers=args.workers,
                                    is_training=True, img_size=args.img_size,
                                    auto_augment=args.aa, distributed=args.distributed)
        loader_eval = create_loader(data=args.data, name='val', batch_size=args.val_batch_size,
                                    num_workers=args.val_workers,
                                    is_training=False, img_size=args.img_size, 
                                    auto_augment=None, distributed=args.distributed)
        loader_de = create_loader(data=args.data, name='train2p', batch_size=args.de_batch_size,
                                    num_workers=args.val_workers,
                                    is_training=False, img_size=args.img_size,
                                    auto_augment=None, distributed=args.distributed)
    if args.num_classes == 100:
        print("create dataset: cifar100")
        loader_train,loader_eval,loader_de = create_loader_cifar(args)
    return loader_train, loader_eval, loader_de



