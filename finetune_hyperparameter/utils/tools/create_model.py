import utils.snn_model.Res19 as Res19
from utils.snn_model import SEW, spiking_resnet
import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from datetime import datetime
from utils.tools.utility import *
import shutil
from utils.tools.option import args_text
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from math import cos, pi
def create_model(args):
    # model setting;    
    if args.model.split('_')[0].lower() == 'sew':
        if args.model.split('_')[1].lower() == '18':
            model = SEW.resnet18(num_classes=args.num_classes, g="add", down='max', T=args.T)
        elif args.model.split('_')[1].lower() == '34':
            model = SEW.resnet34(num_classes=args.num_classes, g="add", down='max', T=args.T)
        else:
            print("Unknown model!!!!")
    elif args.model.split('_')[0].lower() == 'spiking':
        model = spiking_resnet.__dict__[args.model](zero_init_residual=args.zero_init_residual, T=args.T)
    elif args.model.split('_')[0].lower() == 'resnet19':
        # model = ResNet19(num_classes=args.num_classes, total_timestep=args.T)
        model = Res19.resnet19_(num_classes=args.num_classes, T=args.T)
    else:
        print("Unknow model!!!")
        return
    if args.resume: 
        checkpoint = torch.load(args.resume, map_location='cpu')
        if "model" in checkpoint:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.load_state_dict(checkpoint['model'],strict=False)
            else:
                model.load_state_dict(checkpoint['model'],strict=False)
        elif "state_dict" in checkpoint:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.load_state_dict(checkpoint['state_dict'],strict=False)
            else:
                model.load_state_dict(checkpoint['state_dict'],strict=False)
        else:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.load_state_dict(checkpoint,strict=False)
            else:
                model.load_state_dict(checkpoint,strict=False)
    # if args.change_last_layer:
    new_fc = nn.Linear(in_features=model.fc.in_features, out_features=100)
    model.fc = new_fc
    if args.distributed:
       model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
       model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
    return model

def create_optimizer(args,model_parameters):
    if args.opt == 'sgd':
        print("SGD!!!")
        # optimizer = torch.optim.SGD(model_parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        # optimizer = torch.optim.SGD(model_parameters, args.lr, 0.9, args.weight_decay, nesterov=True)
        optimizer = torch.optim.SGD(params=model_parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model_parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        print("Unknow optimizer")
        return
    return optimizer

    
def create_dir_copy_file(args,current_dir):
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), args.exp_name])
        output_dir = get_outdir(output_base, 'train', exp_name, inc=True)
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        shutil.copytree(os.path.join(current_dir, 'utils'), os.path.join(output_dir, 'utils'))
        for filename in os.listdir(current_dir):
            if filename.endswith('.py') or filename.endswith('.sh'):
                src_path = os.path.join(current_dir, filename)
                dst_path = os.path.join(output_dir, filename)
                shutil.copy(src_path, dst_path)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
        tb_writer = SummaryWriter(output_dir + '/_logs')
    
    return tb_writer,output_dir

def plot_with_tb(tb_writer,eval_metrics,loss_all_list,epoch,T):
    tb_writer.add_scalar('top1', eval_metrics['top1'], epoch)
    tb_writer.add_scalar('top5', eval_metrics['top5'], epoch)
    tb_writer.add_scalar('loss', eval_metrics['loss'], epoch)
    loss_all_tb = {}
    for i in range(T):
        print("loss ",i,":",loss_all_list[i],epoch)
        loss_all_tb['loss_different_'+str(i)]=loss_all_list[i]
    print(loss_all_tb)
    tb_writer.add_scalars('loss_different_T', loss_all_tb, global_step=epoch)

def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min, lr_max, warmup=True):
    warmup_epoch = 5 if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max - lr_min) * (
                    1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr