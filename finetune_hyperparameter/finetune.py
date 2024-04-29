import os
import time
import shutil
import logging
from datetime import datetime
from itertools import islice
from collections import OrderedDict
from contextlib import suppress
from spikingjelly.clock_driven import functional
# from utils.models import resume_checkpoint
from utils.tools.utility import *
from utils.tools.option import args, args_text, loss_scaler, amp_autocast, obtain_loader
from utils.tools.common import cosine_lr, mixup_data,mixup_criterion, LabelSmoothingCrossEntropy 
import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.utils.tensorboard import SummaryWriter
from utils.tools.functions import TET_loss,cal_loss_with_ratio
from utils.tools.create_model import create_model,create_optimizer,create_dir_copy_file,plot_with_tb,adjust_learning_rate
# from cifar_finetune.utils.data.cifar_loader_1 import split_cifar100
_logger = logging.getLogger('train')

def main():
    setup_default_logging(log_path=args.log_dir)
    logging.basicConfig(level=logging.DEBUG, filename=args.log_dir, filemode='a')
    random_seed(args.seed, args.rank)

    # dataloader setting;    
    loader_train, loader_eval = obtain_loader(args)
    # loader_train_list, loader_eval = split_cifar100(args)
    # loader_train = loader_train_list[args.No_dataset]
    # model setting;    
    model = create_model(args)
    # for name, param in model.named_parameters():
    #     if not name.startswith('fc'):
    #         param.requires_grad = False
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.local_rank == 0:
        _logger.info(f"Creating model..\n, number of params: {n_parameters}")
    model.cuda()
    model = model.to(memory_format=torch.channels_last)
    # optimizer setting;    
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = create_optimizer(args,model_parameters)
    # scheduler setting;    
    # num_batches = args.slice_len or len(loader_train)
    # scheduler = cosine_lr(optimizer, args.lr, 5*num_batches, args.epochs * num_batches,args.lr_min)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.epochs), eta_min= 0)

    if args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = torch.nn.CrossEntropyLoss().cuda()
    validate_loss_fn = torch.nn.CrossEntropyLoss().cuda()
    tb_writer = None
    tb_writer,args.output_dir = create_dir_copy_file(args,os.path.dirname(os.path.abspath(__file__)))
    #only_test
    if args.only_test:
        eval_metrics,_ = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
        print("Top1: ",eval_metrics['top1'])
        return
    #Main loop
    for epoch in range(args.epochs):
        if args.distributed:
            loader_train.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, max_epoch=args.epochs, lr_min=args.lr_min, lr_max=args.lr, warmup=False)
        train_metrics = train_epoch(
            epoch, model, loader_train, optimizer, train_loss_fn, args,
            None, amp_autocast, loss_scaler, tb_writer)
        # scheduler.step()
        if args.distributed:
            if args.local_rank == 0: _logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, True)
        eval_metrics,loss_all_list = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)\

        if tb_writer is not None: 
            print("tb")
            plot_with_tb(tb_writer,eval_metrics,loss_all_list,epoch,args.T)

        if args.local_rank == 0:
            update_summary(epoch, train_metrics, eval_metrics, os.path.join(args.output_dir, 'finetune.csv'), write_header=True)
            print("Top1: ",eval_metrics['top1'])
            if args.epochs-10 <= epoch <= args.epochs-1 or epoch%50 == 0:
                model_path = os.path.join(args.output_dir, f'{args.exp_name}_{epoch}.pt')
                print('Saving model to', model_path)
                torch.save(model.state_dict(), model_path)

def train_epoch(epoch, model, loader, optimizer, loss_fn, args,
	    lr_scheduler=None, amp_autocast=suppress, loss_scaler=None, tb_writer=None):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    model.train()
    end = time.time()
    slice_len = args.slice_len or len(loader)
    last_idx = slice_len - 1

    for batch_idx, (input, target) in enumerate(islice(loader, slice_len)):
        data_time_m.update(time.time() - end)
        input, target = input.cuda(), target.cuda()
        # input, target = input.cuda(), target.cuda()
        num_updates = batch_idx + epoch * slice_len
        # lr_scheduler(num_updates)
        optimizer.zero_grad()
        # batch = maybe_dictionarize_batch(batch)
        # input, target = batch['images'].cuda(), batch['labels'].cuda()
        # input = input.contiguous(memory_format=torch.channels_last) #!!!
        if args.mixup:
            input, target_a, target_b, lam = mixup_data(input, target, beta=args.mixup)
        with amp_autocast():
            output,output_list = model(input) 
            if args.mixup: 
                loss = mixup_criterion(loss_fn, output, target_a, target_b, lam)
            else: 
                if args.TET:
                    print("TET loss!!!!")
                    loss = TET_loss(output_list.permute(1, 0, 2), target, loss_fn, args.means, args.lamb)#!!!!
                elif args.cal_ratio_loss:
                    time_ratio = torch.tensor(args.time_ratio, dtype=torch.float32)
                    loss = cal_loss_with_ratio(output_list.permute(1, 0, 2),target,loss_fn,time_ratio)
                else:
                    loss = loss_fn(output, target)
                    
        if loss_scaler is not None:
            loss_scaler(loss, optimizer)
        else:
            loss.backward()
            optimizer.step()
        functional.reset_net(model)

        torch.cuda.synchronize()
        batch_time_m.update(time.time() - end)

        if args.local_rank == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            if tb_writer is not None: tb_writer.add_scalar('lr', lr, num_updates)
        
        if (batch_idx == last_idx) or batch_idx % args.log_interval == 0:
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
            else:
                losses_m.update(loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info('Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))
        end = time.time()
    return OrderedDict([('loss', losses_m.avg)])

def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    loss_all_list = [AverageMeter() for i in range(args.T)]
    model.eval()
    end = time.time()
    slice_len = args.slice_len or len(loader)
    last_idx = slice_len - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(islice(loader, slice_len)):
            data_time_m.update(time.time() - end)
            last_batch = batch_idx == last_idx
            # input, target = input.to(torch.float16).cuda(), target.cuda()
            input, target = input.cuda(), target.cuda()

            input = input.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                output_mean,output_list = model(input)
            # mean_out = torch.mean(output, dim=1) #!!!!
            # C,T,N = output.shape
            # mean_out = torch.zeros(C,N).cuda()
            # loss_ratio = torch.tensor(args.loss_ratio, dtype=torch.float32)
            # for t in range(T):
            #     mean_out += loss_ratio[t]*output[:, t, ...]
            loss = loss_fn(output_mean, target)
            loss_all_temp = [loss_fn(i, target) for i in output_list]
            acc1, acc5 = accuracy(output_mean, target, topk=(1, 5))
            functional.reset_net(model)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
                reduced_loss_all_temp = [reduce_tensor(i.data, args.world_size) for i in loss_all_temp]
            else:
                reduced_loss = loss.data
                reduced_loss_all_temp = [i.data for i in loss_all_temp]

            torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), input.size(0))
            top5_m.update(acc5.item(), input.size(0))
            for i in range(len(loss_all_list)):
                loss_all_list[i].update(reduced_loss_all_temp[i].item(), input.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m, data_time=data_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))
                
    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics,[i.avg for i in loss_all_list]

if __name__ == '__main__':
    main()
# import pdb; pdb.set_trace()
