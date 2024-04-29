import os
import time
import shutil
import logging
from datetime import datetime
from collections import OrderedDict
from contextlib import suppress
from itertools import islice
import numpy as np
from random import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from utils.snn_model import SEW
import math
from utils.data import population_info as pop_info
from utils.models import create_model, load_checkpoint
from utils.tools import resume
from utils.tools.utility import *
from utils.tools.option_de import args, args_text, amp_autocast, obtain_loader
from utils.tools import spe
from utils.tools.de import de
from utils.tools.spe import model_dict_to_vector, model_vector_to_dict
# from utils.tools.plot_utils import plot_loss, plot_paras
# from torch.utils.tensorboard import SummaryWriter
# import wandb
from utils.tools.plot_utils_ import plot_top1_vs_baseline
from utils.tools import val
from utils.tools.greedy_soup_ann import greedy_soup,test_ood,test_single_model_ood
from spikingjelly.clock_driven import functional

_logger = logging.getLogger('train')
def main():
    os.environ['WANDB_MODE'] = 'offline'
    # wandb.init(
    #     project='spe',
    #     name='gd',
    #     entity = 'spe_gd',
    #     config = args,
    # )
    setup_default_logging(log_path=args.log_dir)
    logging.basicConfig(level=logging.DEBUG, filename=args.log_dir, filemode='a')
    random_seed(args.seed, args.rank)
    
    # dataloader setting;    
    _, loader_eval, loader_de = obtain_loader(args)
    
    # model setting;
    # model = create_model(args.model, pretrained=False, drop_rate=0.)
    model = SEW.resnet34(num_classes=100, g="add", down='max', T=4)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.local_rank == 0:
        _logger.info(f"Creating model...{args.model},\n number of params: {n_parameters}")
    model.cuda()
    model = model.to(memory_format=torch.channels_last)

    load_score = False
    if os.path.basename(args.pop_init).split('_')[-1] == 'score.txt':
        load_score = True
        score, acc1, acc5, val_loss, en_metrics, models_path = pop_info.get_path_with_acc(args.pop_init)
    else:
        models_path = pop_info.get_path(args.pop_init)
     # ---------- Methods of choosing parent-------------# 
    # optionally resume from a checkpoint
    population = []
    for resume_path in models_path:
        print(resume_path)
        resume.load_checkpoint(model, resume_path, log_info=args.local_rank == 0)
        solution = model_dict_to_vector(model).detach()
        population.append(solution)

    if args.distributed:
       model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
       model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1

    # output_dir = ''
    # tb_writer = None
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), args.exp_name])
        output_dir = get_outdir(output_base, 'train', exp_name, inc=True)
        args.output_dir = output_dir
        current_dir = os.path.dirname(os.path.abspath(__file__))
        shutil.copytree(os.path.join(current_dir, 'utils'), os.path.join(output_dir, 'utils'))
        for filename in os.listdir(current_dir):
            if filename.endswith('.py') or filename.endswith('.sh'):
                src_path = os.path.join(current_dir, filename)
                dst_path = os.path.join(output_dir, filename)
                shutil.copy(src_path, dst_path)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
        # tb_writer = SummaryWriter(output_dir + '/_logs')

    args.popsize = len(models_path)
    if not load_score:
        score = score_func(model, population, loader_de, args) 
        if args.test_ood:
            greedy_model_dict = greedy_soup(population, score, model, loader_de, args,amp_autocast = amp_autocast)
            model.load_state_dict(greedy_model_dict)
            greedy_metrics = val.validate(model, loader_eval, args, amp_autocast=amp_autocast)
            ood_metrics=test_ood(greedy_model_dict,args,model, population, args.popsize, amp_autocast=amp_autocast)
        acc1, acc5, val_loss = [torch.zeros(args.popsize).tolist() for _ in range(3)]
        for i in range(args.popsize): 
            solution = population[i]
            model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
            model.load_state_dict(model_weights_dict)
            temp = val.validate(model, loader_eval, args, amp_autocast=amp_autocast)
            acc1[i], acc5[i], val_loss[i] = temp['top1'], temp['top5'], temp['loss']      
        en_metrics = val.validate_ensemble(model, population, args.popsize, loader_eval, args, amp_autocast=amp_autocast)
        # import pdb; pdb.set_trace()
        # print(score, acc1, acc5, val_loss, en_metrics, models_path)
        pop_info.write_path_with_acc(score, acc1, acc5, val_loss, en_metrics, models_path, args.pop_init)

    # print(score, acc1, acc5, val_loss, en_metrics, models_path)
    if args.local_rank == 0:
        update_summary('baselines:', OrderedDict(en_metrics), os.path.join(output_dir, 'summary.csv'), write_header=True)
        if args.test_ood:
            update_summary('greedy_val:', greedy_metrics, os.path.join(output_dir, 'summary.csv'), write_header=True)
            update_summary('greedy_and_ensemble_ood:', ood_metrics, os.path.join(output_dir, 'summary.csv'), write_header=True)
    rowd = OrderedDict([('score', score), ('top1', acc1), ('top5', acc5), ('val_loss', val_loss)])
    # print(score:)[tensor(1.1641, device='cuda:0', dtype=torch.float16), ,,,]
    if args.local_rank == 0:
        bestidx = score.index(max(score))
        _logger.info('epoch:{}, best_score:{:>7.4f}, best_idx:{}, \
                     score: {}'.format(0, max(score), bestidx, score))
        update_summary(0, rowd, os.path.join(output_dir, 'summary.csv'), write_header=True)

    # eval_metrics_ensemble_temp = val.validate_ensemble(model, population, args.popsize, loader_eval, args, amp_autocast=amp_autocast)
    # ***********************************************************************************************************
    # need to initialize in the main
    # population_init = population#copy.deepcopy(population)
    popsize = args.popsize
    max_iters = args.de_epochs
#     memory_size, lp, cr_init, f_init, k_ls = args.shade_mem, args.shade_lp, args.cr_init, args.f_init, [0,0,0,0]
#     dim = len(model_dict_to_vector(model))
#     # Initialize memory of control settings
#     u_f = np.ones((memory_size, 4)) * f_init
#     u_cr = np.ones((memory_size, 4)) * cr_init
#     u_freq = np.ones((memory_size, 4)) * args.freq_init
#     ns_1, nf_1, ns_2, nf_2, dyn_list_nsf = [], [], [], [], []
#     stra_perc = (1-args.trig_perc)/4
# #     p1_c, p2_c, p3_c, p4_c, p5_c = 1,0,0,0,0
#     p1_c, p2_c, p3_c, p4_c, p5_c = stra_perc, stra_perc, stra_perc, stra_perc, args.trig_perc
#     succ_ls = np.zeros([4, 13])
#     # ---------- set the vector for trainable parameter in DE-------------#   
#     train_bool = torch.from_numpy(np.array([True]*population[0].numel())).cuda()
#     paras1 = [lp, cr_init, f_init, dim, popsize, max_iters, train_bool]
#     paras2 = [p1_c, p2_c, p3_c, p4_c, p5_c, ns_1, nf_1, ns_2, nf_2, u_freq, u_f, u_cr, k_ls, dyn_list_nsf, succ_ls] 
    # plot
    #____________store the inital result in different file_________
    # ***********************************************************************************************************
    print("score",score)
    f = args.f_init
    cr = args.cr_init
    f_cr_threshold = 0
    max_acc_train=0
    max_acc_val=0
    for epoch in range(1, args.de_epochs):
        epoch_time = AverageMeter()
        end = time.time()
        # evolve_out = spe.evolve(score_func, epoch, population, score, paras1, paras2, model, loader_de, args)
        print("cr f",cr,f)
        population,update_label,score = de(popsize, f, cr, population,model,loader_de,args)
        # # -------- cr f change stratgy 1
        # if update_label.count(1) > 0:
        #     f_cr_threshold == 3
        # if update_label.count(1) == 0:
        #     if f_cr_threshold<=0:            
        #         f = args.fcr_min + (args.f - 0.000001) *(1 + math.cos(math.pi * epoch / 8)) / 2 
        #         cr = args.fcr_min + (args.cr - 0.000001) *(1 + math.cos(math.pi * epoch / 8)) / 2 
        #     else:
        #         f_cr_threshold-=1

        # # -------- cr f change 

        # -------- cr f change stratgy 2
        # fcr_min = 1e-9
        # f = fcr_min + (args.f_init - fcr_min) *(1 + math.cos(math.pi * epoch / 40)) / 2 
        # cr =fcr_min + (args.cr_init - fcr_min) *(1 + math.cos(math.pi * epoch / 40)) / 2 


        # -------- cr f change 
        # # -------- cr f change stratgy 3
        fcr_min = 1e-6
        f = fcr_min + (args.f_init - fcr_min) *(1 + math.cos(math.pi * epoch / 40)) / 2 +random.uniform(args.f_init, fcr_min)
        cr =fcr_min + (args.cr_init - fcr_min) *(1 + math.cos(math.pi * epoch / 40)) / 2 +random.uniform(args.cr_init, fcr_min)

        # # -------- cr f change 
        # # -------- cr f change stratgy 4
        # if update_label.count(1) >1:
        #     f_cr_threshold+=1
        # else:
        #     fcr_min = 0.000000001
        #     f = fcr_min + (args.f_init - fcr_min) *(1 + math.cos(math.pi * (epoch-f_cr_threshold) / 5)) / 2 +random.uniform(args.f_init, fcr_min)
        #     cr =fcr_min + (args.cr_init - fcr_min) *(1 + math.cos(math.pi * (epoch-f_cr_threshold) / 5)) / 2 +random.uniform(args.cr_init, fcr_min)
        #     f_cr_threshold=0
        # # -------- cr f change 
        # # -------- cr f change stratgy 5
        # fcr_min = 1e-9
        # f = fcr_min + (args.f_init - fcr_min) *(math.cos(math.pi * epoch / 5)) / 2 # +random.uniform(args.f_init, fcr_min)
        # cr =fcr_min + (args.cr_init - fcr_min) *(math.cos(math.pi * epoch / 5)) / 2 # +random.uniform(args.cr_init, fcr_min)

        # # -------- cr f change 
        if args.local_rank == 0:
            # population, score, bestidx, worstidx, dist_matrics, paras2, update_label = evolve_out
            # p1_c, p2_c, p3_c, p4_c, p5_c, ns_1, nf_1, ns_2, nf_2, u_freq, u_f, u_cr, k_ls, dyn_list_nsf, succ_ls = paras2
            bestidx = score.index(max(score))

            _logger.info('epoch:{}, best_score:{:>7.4f}, best_idx:{}, \
                     score: {}'.format(0, max(score), bestidx, score))
            pop_tensor = torch.stack(population)
            if max(score)>max_acc_train:
                max_acc_train = max(score)
                print("Best in train_set update and train acc = ",max_acc_train)
                model_path = os.path.join(output_dir, f'train_best_{args.model}.pt')
                print('Saving model to', model_path)
                torch.save(model.state_dict(), model_path)


        if args.local_rank != 0: 
            update_label = list(range(popsize))
            pop_tensor = torch.stack(population)

        if args.distributed: 
            torch.cuda.synchronize()
            dist.barrier() 
            torch.distributed.broadcast_object_list(update_label, src=0)
            torch.distributed.broadcast(pop_tensor, src=0)

        population = list(pop_tensor)
        for i in range(popsize): #!!!
            if update_label[i] == 1:
                solution = population[i]
                model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
                model.load_state_dict(model_weights_dict)
                temp = val.validate(model, loader_eval, args, amp_autocast=amp_autocast)
                acc1[i], acc5[i], val_loss[i] = temp['top1'], temp['top5'], temp['loss'] 
                if acc1[i]>max_acc_val:
                    max_acc_val = acc1[i]
                    print("Best in train_set update and val acc = ",max_acc_val)
                    model_path = os.path.join(output_dir, f'val_best_{args.model}.pt')
                    print('Saving best val model to', model_path)
                    torch.save(model.state_dict(), model_path)

        if args.distributed: 
            torch.cuda.synchronize()
        epoch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0:
            _logger.info('score: {}'.format(rowd['score']))
            _logger.info('DE:{} Acc@1: {top1:>7.4f} Acc@5: {top5:>7.4f} \
                         Epoch_time: {epoch_time.val:.3f}s'.format(
                            epoch,
                            top1 = rowd['top1'][bestidx],
                            top5 = rowd['top5'][bestidx],
                            epoch_time=epoch_time))
            rowd = OrderedDict([('best_idx',bestidx),('score', score), ('top1', acc1), ('top5', acc5), ('val_loss', val_loss)])
        
            update_summary(epoch, rowd, os.path.join(output_dir, 'summary.csv'), write_header=True)
            bestidx_tensor = torch.tensor(bestidx).cuda()
            if args.de_epochs-popsize <= epoch <= args.de_epochs-1:
                model_path = os.path.join(args.output_dir, f'{args.exp_name}_{epoch}.pt')
                print('Saving model to', model_path)
                torch.save(model.state_dict(), model_path)
            # if not args.test_ood:
            #     plot_top1_vs_baseline(output_dir,'./result',exp_name,None)
        # if args.test_ood:
        #     if args.local_rank != 0: 
        #         bestidx_tensor=torch.tensor(0).cuda()
        #     if args.distributed: 
        #         torch.distributed.broadcast(bestidx_tensor, src=0)
        #     solution = population[bestidx_tensor]
        #     model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
        #     model.load_state_dict(model_weights_dict)
        #     de_ood_metric = test_single_model_ood(args,model)
        #     if args.local_rank == 0:
        #         update_summary('de_ood:', de_ood_metric, os.path.join(output_dir, 'summary.csv'), write_header=True)

            # plot_loss(output_dir, popsize, wandb)


    wandb.finish()
    return

def score_func(model, population, loader_de, args):
    popsize = len(population)
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc1_all = torch.zeros(popsize).tolist()
    acc5_all = torch.zeros(popsize).tolist()
    end = time.time()
    model.eval()
    torch.set_grad_enabled(False)
    slice_len = args.de_slice_len or len(loader_de)
    for batch_idx, (input, target) in enumerate(islice(loader_de, slice_len)):
        data_time_m.update(time.time() - end)
        for i in range(0, popsize):
            solution = population[i]
            model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
            model.load_state_dict(model_weights_dict)
            input, target = input.cuda(), target.cuda()
            input = input.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                output,_ = model(input)
            # if batch_idx==0 and args.local_rank < 2 and i == 0:
            #     _logger.info('Checking Data >>>> pop: {} input: {}'.format(i, input.flatten()[6000:6005]))
            functional.reset_net(model)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if args.distributed:
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            acc1_all[i] += acc1
            acc5_all[i] += acc5
        batch_time_m.update(time.time() - end)
        end = time.time()

    if args.local_rank == 0:
        print('data_time: {time1.val:.3f} ({time1.avg:.3f})  '
            'batch_time: {time2.val:.3f} ({time2.avg:.3f})  '.format(time1=data_time_m, time2=batch_time_m)) 

    score = [i.cpu()/slice_len for i in acc1_all]#!!!
    return score

if __name__ == '__main__':
    main()
