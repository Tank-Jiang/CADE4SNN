""" 
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import distributed as dist

import random
import numpy as np
import csv
import os
import logging
import logging.handlers
from collections import OrderedDict
try: 
    import wandb
except ImportError:
    pass

""" Summary utilities """
def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 1000
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir

def update_summary(epoch, rowd_in, filename, write_header=True, log_wandb=False):
    rowd = OrderedDict(epoch=epoch)
    for key, value in rowd_in.items():
        if isinstance(value, list):
            if isinstance(value[0], torch.Tensor):
                list_temp = []
                for element in value:
                    list_temp.append(round(element.item()))
                rowd_in[key] = list_temp
    rowd.update(rowd_in)
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header: dw.writeheader()
        dw.writerow(rowd)

def update_summary2(epoch, train_metrics, eval_metrics, filename, write_header=False, log_wandb=False):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)

""" Random_seed """
def random_seed(seed=42, rank=0):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    random.seed(seed + rank)

""" Distributed training/validation utils """
def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model
    # if isinstance(model, ModelEma): return unwrap_model(model.ema)

def distribute_bn(model, world_size, reduce=True):
    # ensure every node has the same running bn stats
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                # average bn stats across whole group
                torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                bn_buf /= float(world_size)
            else:
                # broadcast bn stats from rank 0 to whole group
                torch.distributed.broadcast(bn_buf, 0)

""" CUDA / AMP utils """
class NativeScaler:
    state_dict_key = "amp_scaler"
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        self._scaler.step(optimizer)
        self._scaler.update()
    # def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
    #     self._scaler.scale(loss).backward(create_graph=create_graph)
    #     if clip_grad is not None:
    #         assert parameters is not None
    #         self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
    #         dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
    #     self._scaler.step(optimizer)
    #     self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

""" Logging helpers """
class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)

def setup_default_logging(default_level=logging.INFO, log_path=''):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)

""" Eval metrics and related """
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
