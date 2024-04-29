# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Evolving weights training and evaluating script
This script is modified from pytorch-image-models by Ross Wightman (https://github.com/rwightman/pytorch-image-models/)
It was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)
"""
import os
import time
import yaml
import shutil
import logging
import numpy as np
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import wandb
# import matplotlib.pyplot as plt
# import models
import random

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from utils import shade
from utils.tools.shade import model_dict_to_vector, model_vector_to_dict
from utils.tools.plot_utils import plot_loss, plot_paras
from utils.data import create_loader, resolve_data_config
from timm.models import create_model, resume_checkpoint
# from timm.utils import *
# from timm.utils import ApexScaler, NativeScaler
from timm.utils import reduce_tensor, AverageMeter, update_summary, setup_default_logging, get_outdir, accuracy
from utils.tools.option import get_loader_args_and_dataset
import copy

# import vit_snn
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
_logger = logging.getLogger('train')

def main():
    os.environ['WANDB_MODE'] = 'offline'
    from utils.tools.option import args, args_text, amp_autocast  #
    # loss_scaler = NativeScaler()
    # amp_autocast = torch.cuda.amp.autocast
    wandb.init(
        project='evo_ann_gd',
        name='-1',
        entity = 'evolving',
        config = args,
    )
    setup_default_logging(log_path=args.log_dir)
    logging.basicConfig(level=logging.DEBUG, filename=args.log_dir, filemode='a')

    torch.manual_seed(args.seed + args.rank)
    # random.seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)
    torch.cuda.manual_seed_all(args.seed + args.rank)
#     import pdb; pdb.set_trace()
    model = create_model(args.model, pretrained=False, drop_rate=0.)
    print("Creating model")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    if args.local_rank == 0:
        _logger.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    model.cuda()
    if args.channels_last:
       model = model.to(memory_format=torch.channels_last)

    # ---------- Methods of choosing parent-------------# 
    # optionally resume from a checkpoint
    if args.resume_parents:
        load_pop = torch.load(args.parent_pop_dir, map_location='cpu')
        population = list(load_pop['pop'].cuda())
    else:
        population = []
        for file in os.listdir(args.pop_init_dir):
           if len(population) >= args.popsize: break
           else: 
              if file.split('-')[0] == 'checkpoint':
                   resume_path = os.path.join(args.pop_init_dir, file)
                   resume_epoch = resume_checkpoint(model, resume_path, log_info=args.local_rank==0)-1
                   solution = model_dict_to_vector(model).detach()
                   population.append(solution)

    if args.noise_parent:
        pop_std = torch.std(torch.stack(population), dim=0)
        for i in range(1, args.popsize):
            population[i] = torch.normal(population[0], pop_std)

    # ---------- Methods of choosing parent-------------# 


    if args.distributed and args.sync_bn:
       assert not args.split_bn
       model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
       model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    #   set the vector for trainable parameter in DE
    if args.restrict_para:
        train_bool = []
        fix_para, unfix_para = 0, 0
        for key, curr_weights in model.state_dict().items():
            if not('fc.' in key or 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key):
            #if not ('running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key):
            # if 'weight' in key and 'conv1' in key:
            # if 'layer4' in key and not ('running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key):
                train_bool.append([True]*curr_weights.numel())
                unfix_para += curr_weights.numel()
            else:
                train_bool.append([False]*curr_weights.numel())
                fix_para += curr_weights.numel()
        train_bool = torch.from_numpy(np.concatenate(train_bool)).cuda()
        if args.local_rank == 0:
            _logger.info('fix %d unfix %d' %(fix_para, unfix_para))
    else:
        train_bool = torch.from_numpy(np.array([True]*population[0].numel())).cuda()

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
    loader_de_args, dataset_de, loader_val_args, dataset_eval = get_loader_args_and_dataset(args, data_config)
    loader_eval = create_loader(dataset_eval, **loader_val_args)
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    
    if args.only_test:
       val_metrics = validate(model, loader_eval, validate_loss_fn, args)
       _logger.info(f"Top-1,5 accuracy of the model is: {val_metrics['top1']:.3f}%, {val_metrics['top5']:.3f}%")
       return
        
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model,
            str(data_config['input_size'][-1])
        ])
        output_dir = get_outdir(output_base, 'train', exp_name)
        args.output_dir = output_dir
        current_dir = os.path.dirname(os.path.abspath(__file__))
        shutil.copytree(os.path.join(current_dir, 'utils'), os.path.join(output_dir, 'utils'))
        for filename in os.listdir(current_dir):
            if filename.endswith('.py') or filename.endswith('.sh'):
                src_path = os.path.join(current_dir, filename)
                dst_path = os.path.join(output_dir, filename)
                shutil.copy(src_path, dst_path)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f: f.write(args_text)
       
    eval_metrics_acc1 = torch.zeros(args.popsize).tolist()
    eval_metrics_acc5 = torch.zeros(args.popsize).tolist()
    eval_metrics_loss = torch.zeros(args.popsize).tolist()
    for i in range(args.popsize): #!!!
         solution = population[i]
         model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
         model.load_state_dict(model_weights_dict)
         if args.skip_init_validate: #for fast val when debugging
              eval_metrics_temp = OrderedDict([('top1', 233), ('top5', 233),('loss', 233)])
         else:
              eval_metrics_temp = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
         eval_metrics_acc1[i] = round(eval_metrics_temp['top1'], 4)
         eval_metrics_acc5[i] = round(eval_metrics_temp['top5'], 4)
         eval_metrics_loss[i] = round(eval_metrics_temp['loss'], 4) 
            
    eval_metrics = OrderedDict([('top1', eval_metrics_acc1), ('top5', eval_metrics_acc5), ('eval_loss', eval_metrics_loss)])

    eval_metrics_ensemble_temp = validate_ensemble(model, population,args.popsize, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
    eval_metrics['ensemble_top1'] = round(eval_metrics_ensemble_temp['top1'], 4)
    eval_metrics['ensemble_top5'] = round(eval_metrics_ensemble_temp['top5'], 4)
    eval_metrics['ensemble_eval_loss'] = round(eval_metrics_ensemble_temp['loss'], 4)
    eval_metrics['ensemble_top1_sm'] = round(eval_metrics_ensemble_temp['top1_sm'], 4)
    eval_metrics['ensemble_top5_sm'] = round(eval_metrics_ensemble_temp['top5_sm'], 4)
    eval_metrics['ensemble_eval_loss_sm'] = round(eval_metrics_ensemble_temp['loss_sm'], 4)
    eval_metrics['ensemble_top1_mv'] = round(eval_metrics_ensemble_temp['top1_mv'], 4)
    eval_metrics['ensemble_top1_ema'] = round(eval_metrics_ensemble_temp['top1_ema'], 4)
    eval_metrics['ensemble_top5_ema'] = round(eval_metrics_ensemble_temp['top5_ema'], 4)
    eval_metrics['ensemble_eval_loss_ema'] = round(eval_metrics_ensemble_temp['loss_ema'], 4)
    # ***********************************************************************************************************
    # need to initialize in the main
    # population_init = population#copy.deepcopy(population)
    popsize = args.popsize
    max_iters = args.de_iters
    memory_size, bounds, lp, cr_init, f_init, k_ls = args.shade_mem, args.bounds, args.shade_lp, args.cr_init, args.f_init, [0,0,0,0]
    dim = len(model_dict_to_vector(model))
    # Initialize memory of control settings
    u_f = np.ones((memory_size,4)) * f_init
    u_cr = np.ones((memory_size,4)) * cr_init
    u_freq = np.ones((memory_size,4)) * args.freq_init
    ns_1, nf_1, ns_2, nf_2, dyn_list_nsf = [], [], [], [], []
    stra_perc = (1-args.trig_perc)/4
#     p1_c, p2_c, p3_c, p4_c, p5_c = 1,0,0,0,0
    p1_c, p2_c, p3_c, p4_c, p5_c = stra_perc, stra_perc, stra_perc, stra_perc, args.trig_perc
    succ_ls = np.zeros([4,13])
    paras1 = [lp, cr_init, f_init, bounds, dim, popsize, max_iters, train_bool]
    paras2 = [p1_c, p2_c, p3_c, p4_c, p5_c, ns_1, nf_1, ns_2, nf_2, u_freq, u_f, u_cr, k_ls, dyn_list_nsf, succ_ls] 
    # plot
    plot_variables = [u_freq_mean, u_f_mean, u_cr_mean, epoch_num, epoch_num_2, \
                        cons_sim_list, l2_dist_list, lowest_dist_list, mean_dist_list, largest_dist_list, \
                        L1_value_list, L2_value_list, succ_ls_list, p_list] = [[] for i in range(14)]
    top_acc = 0
    replace_pop = False
#________________store the inital result in different file_________
    # ***********************************************************************************************************
    for epoch in range(args.de_epochs):
          loader_de_args['worker_seeding'] = epoch + 233
          loader_de = create_loader(dataset_de, **loader_de_args)
          if args.distributed:
               loader_de.sampler.set_epoch(epoch)
          dist.barrier()
          score_lst = score_func(model, population, loader_de, args) 
          # print(score_lst:)[tensor(1.1641, device='cuda:0', dtype=torch.float16), ,,,]
          if args.local_rank == 0:
               bestidx = score_lst.index(min(score_lst))
               worstidx = score_lst.index(max(score_lst))
               de_iter_loss = [round(j.item(), 4) for j in score_lst]
               _logger.info('de_iter:{}, best_score:{:>7.4f}, best_idx:{}, de_iter_loss: {}'.format(
                                                                 0, min(score_lst), bestidx, de_iter_loss))
               de_iter_dict = OrderedDict([('iter', 0), ('bestidx', bestidx), ('worstidx', worstidx), ('train_loss', de_iter_loss)])
               update_summary(epoch, de_iter_dict, eval_metrics, os.path.join(output_dir, 'summary.csv'), write_header=True)

          de_iter_time_m = AverageMeter()
          end = time.time()
          for de_iter in range(1, args.de_iters+1):
               # import pdb; pdb.set_trace()               
               evolve_out = shade.evolve(score_func, epoch, de_iter, population, score_lst, paras1, paras2, model, loader_de, args)
               # evolve_out = de.evolve(score_func, epoch, de_iter, population, score_lst, paras1, paras2, model, loader_de, args)
               # evolve_out = sade.evolve(score_func, epoch, de_iter, population, score_lst, paras1, paras2, model, loader_de, args)
               if args.local_rank == 0:
                    population, score_lst, bestidx, worstidx, dist_matrics, paras2, update_label = evolve_out
                    p1_c, p2_c, p3_c, p4_c, p5_c, ns_1, nf_1, ns_2, nf_2, u_freq, u_f, u_cr, k_ls, dyn_list_nsf, succ_ls = paras2
                    cons_sim, l2_dist, lowest_dist, mean_dist, largest_dist, L1_value, L2_value = dist_matrics
                    best_score = score_lst[bestidx]
                    de_iter_loss = [round(j.item(), 4) for j in score_lst]
                    _logger.info('de_iter:{}, best_score:{:>7.4f}, best_idx:{}, worst_idx:{}, de_iter_loss: {}'.format(
                                   de_iter, best_score, bestidx, worstidx, de_iter_loss))
                    pop_tensor = torch.stack(population)

                    de_iter_dict = OrderedDict([('iter', de_iter), ('bestidx', bestidx), ('worstidx', worstidx), 
                                             ('train_loss', de_iter_loss)])
                    parameter1=OrderedDict([('cons_sim', cons_sim), ('l2_dist', l2_dist), 
                                             ('lowest_dist', lowest_dist), ('mean_dist', mean_dist), ('largest_dist', largest_dist)])
                    parameter2=OrderedDict([('u_freq', u_freq), ('u_f', u_f), ('u_cr', u_cr), ('L1_value', L1_value), ('L2_value', L2_value),
                                            ('strat_no', succ_ls[0,:]), ('succ_perc', succ_ls[1,:]/(succ_ls[0,:]+1)), ('strat_no_1', succ_ls[2,:]),
                                             ('succ_perc_1', succ_ls[3,:]/(succ_ls[2,:]+1)) ])
                    update_summary(epoch, parameter1, parameter2, os.path.join(output_dir, 'summary2.csv'), write_header=True)

                    u_freq_mean.append(np.array([u_freq[k_ls[ik]-1, ik] for ik in range(4)]))
                    u_f_mean.append(np.array([u_f[k_ls[ik]-1, ik] for ik in range(4)]))
                    u_cr_mean.append(np.array([u_cr[k_ls[ik]-1, ik] for ik in range(4)]))

                    epoch_num.append([epoch*args.de_iters + de_iter for _ in range(len(np.mean(u_cr, axis=0)))])

                    L1_value_list.append(L1_value.cpu().numpy())
                    L2_value_list.append(L2_value.cpu().numpy())
                    epoch_num_2.append([epoch*args.de_iters + de_iter for _ in range(len(L1_value))])

                    # cons_sim_list.append(cons_sim.item())
                    # l2_dist_list.append(l2_dist.item())
                    cons_sim_list.append(cons_sim)
                    l2_dist_list.append(l2_dist)
                    lowest_dist_list.append(lowest_dist.item())
                    mean_dist_list.append(mean_dist.item())
                    largest_dist_list.append(largest_dist.item())
                    succ_ls_list.append(succ_ls.copy())
                    p_list.append([p1_c,p2_c,p3_c,p4_c,p5_c])

                    plot_paras(epoch, de_iter, max_iters, output_dir, plot_variables, wandb)
                    # de_iter_dict = OrderedDict([('epoch', epoch), ('iter', de_iter)])

                    # parameter3=OrderedDict([('u_freq', u_freq), ('u_f', u_f), ('u_cr', u_cr), ('L1_value', L1_value), ('L2_value', L2_value),
                    #     ('strat_no', succ_ls[0,:]), ('succ_perc', succ_ls[1,:]/(succ_ls[0,:]+1)), ('strat_no_1', succ_ls[2,:]),
                    #      ('succ_perc_1', succ_ls[3,:]/(succ_ls[2,:]+1)) ])
                    # update_summary(epoch, parameter3, os.path.join(output_dir, 'summary_strategy.csv'), write_header=True)

                    # if epoch%10 == 0 and de_iter == 1:
                    #     pop_save = OrderedDict([('epoch', epoch), ('de_iter', de_iter), ('pop', torch.stack(population))])  #torch.stack(population)
                    #     torch.save(pop_save, os.path.join(output_dir, 'pop_save'+'_'+str(epoch)+'_'+str(de_iter)+'_'+str(args.local_rank)+'.pt'))

               if args.local_rank != 0: 
                  update_label = list(range(popsize))
                  pop_tensor = torch.stack(population)
               
               # torch.cuda.synchronize()
               dist.barrier() 
               torch.distributed.broadcast_object_list(update_label, src=0)
               torch.distributed.broadcast(pop_tensor, src=0)
               # torch.distributed.broadcast_object_list([update_label, population], src=0)
               dist.barrier() 

               population = list(pop_tensor)
               if (de_iter == args.de_iters and (epoch % args.validate_interval==0)) or args.validate_every_iter: 
                   if args.skip_pop_validate:
                       eval_metrics_ensemble_temp = OrderedDict([('top1', 233), ('top5', 233),('loss', 233),
                        ('top1_sm', 233), ('top5_sm', 233),('loss_sm', 233),('top1_mv', 233), ('top1_ema', 233),('top5_ema', 233),('loss_ema', 233)])
                   else:
                       eval_metrics_ensemble_temp = validate_ensemble(model, population,popsize, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
                   eval_metrics['ensemble_top1'] = round(eval_metrics_ensemble_temp['top1'], 4)
                   eval_metrics['ensemble_top5'] = round(eval_metrics_ensemble_temp['top5'], 4)
                   eval_metrics['ensemble_eval_loss'] = round(eval_metrics_ensemble_temp['loss'], 4)
                   eval_metrics['ensemble_top1_sm'] = round(eval_metrics_ensemble_temp['top1_sm'], 4)
                   eval_metrics['ensemble_top5_sm'] = round(eval_metrics_ensemble_temp['top5_sm'], 4)
                   eval_metrics['ensemble_eval_loss_sm'] = round(eval_metrics_ensemble_temp['loss_sm'], 4)
                   eval_metrics['ensemble_top1_mv'] = round(eval_metrics_ensemble_temp['top1_mv'], 4)
                   eval_metrics['ensemble_top1_ema'] = round(eval_metrics_ensemble_temp['top1_ema'], 4)
                   eval_metrics['ensemble_top5_ema'] = round(eval_metrics_ensemble_temp['top5_ema'], 4)
                   eval_metrics['ensemble_eval_loss_ema'] = round(eval_metrics_ensemble_temp['loss_ema'], 4)

                   for i in range(popsize): #!!!
                       # if successful upate, or the final iteration of the epoch
                       if update_label[i] == 1 or not args.validate_every_iter:
                            solution = population[i]
                            model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
                            model.load_state_dict(model_weights_dict)
                            if args.skip_pop_validate: #for fast val when debugging
                                 eval_metrics_temp = OrderedDict([('top1', 233), ('top5', 233),('loss', 233)])
                            else:
                                 eval_metrics_temp = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
                            eval_metrics['top1'][i] = round(eval_metrics_temp['top1'], 4)
                            eval_metrics['top5'][i] = round(eval_metrics_temp['top5'], 4)
                            eval_metrics['eval_loss'][i] = round(eval_metrics_temp['loss'], 4)
                            if eval_metrics['top1'][i] > top_acc:
                                replace_pop = True
                                top_acc = eval_metrics['top1'][i]

                   torch.cuda.synchronize()
                   de_iter_time_m.update(time.time() - end)
                   end = time.time()
                   
                   if args.local_rank == 0:
                        pop_save = OrderedDict([('epoch', epoch), ('de_iter', de_iter), ('pop', torch.stack(population))])  #torch.stack(population)
                        src_path = os.path.join(output_dir, 'pop_save_last.pt')
                        dst_path = os.path.join(output_dir, 'pop_save_best.pt')
                        torch.save(pop_save, src_path)
                        if replace_pop:
                              shutil.copy(src_path, dst_path)
                              replace_pop = False

                        _logger.info('eval_metrics_acc1: {}'.format(eval_metrics['top1']))
                        _logger.info('DE: {} [de_iter: {}]  '
                                     'Acc@1: {top1:>7.4f}  '
                                     'Acc@5: {top5:>7.4f}  '
                                     'Iter_time: {de_iter_time.val:.3f}s, {rate:>7.2f}/s  '.format(
                                      epoch, de_iter,
                                      top1 = eval_metrics['top1'][bestidx],
                                      top5 = eval_metrics['top5'][bestidx],
                                      de_iter_time=de_iter_time_m,
                                      rate= args.de_batch_size * args.world_size / de_iter_time_m.val))
                        
                        update_summary(epoch, de_iter_dict, eval_metrics, os.path.join(output_dir, 'summary.csv'), write_header=True)
                        plot_loss(output_dir, popsize, wandb)
               dist.barrier() 
          # evolve_out = shade.replace_pop(population, score_lst, popsize/2)
          # population = evolve_out
          # optionally resume from a checkpoint
        #   if args.replace_pop_after_epoch:
        #     # deprecated, as already not recording population_init
        #       if args.local_rank==0:
        #           sorted_population_f, old_idx =  torch.sort(torch.tensor(score_lst), descending=True)
        #           replace_pop_num = int(args.popsize/2)
        #           #pop_counts = 0
        #           #for ip in range(replace_pop_num):
        #               # population[old_idx[ip]] = population_init[ip]

        #           # for file in os.listdir(args.pop_init_dir):
        #           #    if pop_counts >= replace_pop_num: break
        #           #    else: 
        #           #       if file.split('-')[0] == 'checkpoint':
        #           #            resume_path = os.path.join(args.pop_init_dir, file)
        #           #            resume_epoch = resume_checkpoint(model, resume_path, log_info=args.local_rank==0)-1
        #           #            solution = model_dict_to_vector(model).detach()
        #           #            population[old_idx[pop_counts]] = solution
        #           #            pop_counts = pop_counts + 1
        #           pop_tensor = torch.stack(population)
        #       torch.distributed.broadcast(pop_tensor, src=0)
        #       dist.barrier()
        #       population = list(pop_tensor)
        #       dist.barrier()
            #    wandb.log({"image_acc_loss": wandb.Image(os.path.join(output_dir,'plot.svg'))})
    wandb.finish()
    return

def score_func(model, population, loader_de, args):
     train_loss_fn = nn.CrossEntropyLoss().cuda()
     popsize = len(population)
     batch_time_m = AverageMeter()
     data_time_m = AverageMeter()
     acc1_all = torch.zeros(popsize).tolist()
     acc5_all = torch.zeros(popsize).tolist()
     loss_all = torch.zeros(popsize).tolist()#!!!
     end = time.time()
     model.train()
     torch.set_grad_enabled(False)
     for batch_idx, (input, target) in enumerate(loader_de):
          if batch_idx >= (args.de_batch_size//args.mini_batch_size): break
          data_time_m.update(time.time() - end)
          for i in range(0, popsize):
               solution = population[i]
               model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
               model.load_state_dict(model_weights_dict)
               if not args.prefetcher:
                    input = input.cuda()
                    target = target.cuda()
               if args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)
               if batch_idx==0 and args.local_rank < 2 and i == 0:
                    _logger.info('Checking Data >>>> pop: {} input: {}'.format(i, input.flatten()[6000:6005]))
               amp_autocast = torch.cuda.amp.autocast
               with amp_autocast():
                    output = model(input)
               # pick back the batch norm info
               population[i] = model_dict_to_vector(model).detach()
               
               if isinstance(output, (tuple, list)):
                    output = output[0]
               loss = train_loss_fn(output, target)
               acc1, acc5 = accuracy(output, target, topk=(1, 5))
               if args.distributed:
                    reduced_loss = reduce_tensor(loss.data, args.world_size)
                    acc1 = reduce_tensor(acc1, args.world_size)
                    acc5 = reduce_tensor(acc5, args.world_size)
               # _logger.info('acc1: {}  acc5: {}'.format(acc1, acc5))
               acc1_all[i] += acc1
               acc5_all[i] += acc5
               loss_all[i] += reduced_loss
               loss_all[i] += reduced_loss

          batch_time_m.update(time.time() - end)
          end = time.time()

     if args.local_rank == 0:
          print('data_time: {time1.val:.3f} ({time1.avg:.3f})  '
                'batch_time: {time2.val:.3f} ({time2.avg:.3f})  '.format(time1=data_time_m, time2=batch_time_m)) 

     score_lst_loss = [i.cpu()/(args.de_batch_size//args.mini_batch_size) for i in loss_all]#!!!
     return score_lst_loss

def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # if batch_idx > 1: break
            data_time_m.update(time.time() - end)
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # if batch_idx==0 and args.local_rank == 0:#!!!
            #    _logger.info('validate, input: {}'.format(input.flatten()[6000:6005]))#!!!

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            # reduced_loss_all = torch.zeros(popsize).tolist()
            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

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
    if args.local_rank == 0: _logger.info('metrics_top1: {}'.format(metrics['top1']))
    return metrics

def validate_ensemble(model, pop,popsize, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):

     batch_time_m = AverageMeter()
     data_time_m = AverageMeter()
     losses_m = AverageMeter()
     top1_m = AverageMeter()
     top5_m = AverageMeter()

     losses_m_sm = AverageMeter()
     top1_m_sm = AverageMeter()
     top5_m_sm = AverageMeter()

     top1_m_mv = AverageMeter()

     losses_ema = AverageMeter()
     top1_ema = AverageMeter()
     top5_ema = AverageMeter()

     population = pop
     end = time.time()
     last_idx = len(loader) - 1
     
     pop_avg = torch.mean(torch.stack(population), dim=0)
     
     with torch.no_grad():
          for batch_idx, (input, target) in enumerate(loader):
               # if batch_idx > 1: break
               data_time_m.update(time.time() - end)
               last_batch = batch_idx == last_idx
               if not args.prefetcher:
                    input = input.cuda()
                    target = target.cuda()
               if args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

               # if batch_idx==0 and args.local_rank == 0:#!!!
               #    _logger.info('validate, input: {}'.format(input.flatten()[6000:6005]))#!!!
               pop_ouput = []
               pop_pred = []
               for i in range(popsize): #!!!
                    solution = population[i]
                    model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
                    model.load_state_dict(model_weights_dict)

                    model.eval()
                    with amp_autocast():
                         output = model(input)
                    if isinstance(output, (tuple, list)):
                         output = output[0]
                    # reduced_loss_all = torch.zeros(popsize).tolist()
                    # augmentation reduction
                    reduce_factor = args.tta
                    if reduce_factor > 1:
                         output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                         target = target[0:target.size(0):reduce_factor]

                    pop_ouput.append(output.unsqueeze(0))
                    _, pred = torch.max(output, 1)
                    pop_pred.append(pred.unsqueeze(0))

               #mean of output layer ensemble
               output2 = torch.cat(pop_ouput, dim=0)
               output = torch.mean(output2, dim=0)

               loss = loss_fn(output, target)
               acc1, acc5 = accuracy(output, target, topk=(1, 5))
               
               #mean of softmax normlization and mean
               m = nn.Softmax(dim=2)
               output_sm = m(output2)
               output_sm = torch.mean(output_sm, dim=0)

               loss_sm = loss_fn(output_sm, target)
               acc1_sm, acc5_sm = accuracy(output_sm, target, topk=(1, 5))

               # majority voting
               outputt = torch.cat(pop_pred, dim=0)
               #print(outputt,outputt.mode(0).values,target)
               acc1_mv = (outputt.mode(0).values == target).float().mean()*100.0

               #  weight Averaging
               model_weights_dict = model_vector_to_dict(model=model, weights_vector=pop_avg)
               model.load_state_dict(model_weights_dict)
               model.eval()
               with amp_autocast():
                    output = model(input)
               loss_ema = loss_fn(output, target)
               acc1_ema, acc5_ema = accuracy(output, target, topk=(1, 5))

               if args.distributed:
                    reduced_loss = reduce_tensor(loss.data, args.world_size)
                    acc1 = reduce_tensor(acc1, args.world_size)
                    acc5 = reduce_tensor(acc5, args.world_size)

                    reduced_loss_sm = reduce_tensor(loss_sm.data, args.world_size)
                    acc1_sm = reduce_tensor(acc1_sm, args.world_size)
                    acc5_sm = reduce_tensor(acc5_sm, args.world_size)
                    acc1_mv = reduce_tensor(acc1_mv, args.world_size)

                    reduced_loss_ema = reduce_tensor(loss_ema.data, args.world_size)
                    acc1_ema = reduce_tensor(acc1_ema, args.world_size)
                    acc5_ema = reduce_tensor(acc5_ema, args.world_size)
               else:
                    reduced_loss = loss.data
                    reduced_loss_sm = loss_sm.data

               torch.cuda.synchronize()

               losses_m.update(reduced_loss.item(), input.size(0))
               top1_m.update(acc1.item(), output.size(0))
               top5_m.update(acc5.item(), output.size(0))

               losses_m_sm.update(reduced_loss_sm.item(), input.size(0))
               top1_m_sm.update(acc1_sm.item(), output.size(0))
               top5_m_sm.update(acc5_sm.item(), output.size(0))

               top1_m_mv.update(acc1_mv.item(), output.size(0))

               losses_ema.update(reduced_loss_ema.item(), input.size(0))
               top1_ema.update(acc1_ema.item(), output.size(0))
               top5_ema.update(acc5_ema.item(), output.size(0))

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

     metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg),('loss_sm', losses_m_sm.avg), ('top1_sm', top1_m_sm.avg), ('top5_sm', top5_m_sm.avg), ('top1_mv', top1_m_mv.avg),
                            ('loss_ema', losses_ema.avg), ('top1_ema', top1_ema.avg), ('top5_ema', top5_ema.avg)])
     if args.local_rank == 0: _logger.info('ensemble_metrics_top1: {}'.format(metrics['top1']))
     return metrics

if __name__ == '__main__':
    main()
