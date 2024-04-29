import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
# from scipy.stats import cauchy
from contextlib import suppress
from utils.tools.utility import *
from collections import OrderedDict
from .utility import update_summary
import torch.nn as nn
from .datasets import ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA
from .spe import model_vector_to_dict
from itertools import islice
from spikingjelly.clock_driven import functional
from utils.tools.common import maybe_dictionarize_batch
from utils.data.transform_timm.transforms_factory import transforms_imagenet_train, transforms_imagenet_eval
def test_model(model, loader, args, amp_autocast=suppress):
    losses_avg = AverageMeter()
    top1_avg = AverageMeter()
    top5_avg = AverageMeter()
    loss_fn = nn.CrossEntropyLoss().cuda()
    slice_len = args.slice_len or len(loader)
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # if batch_idx > 1: break
            input = input.cuda()
            target = target.cuda()
            input = input.contiguous(memory_format=torch.channels_last)
            model.eval()
            with amp_autocast():
                output,_ = model(input)            
            loss_avg = loss_fn(output, target)
            acc1_avg, acc5_avg = accuracy(output, target, topk=(1, 5))
            functional.reset_net(model)
            if args.distributed:
                reduced_loss_avg = reduce_tensor(loss_avg.data, args.world_size)
                acc1_avg = reduce_tensor(acc1_avg, args.world_size)
                acc5_avg = reduce_tensor(acc5_avg, args.world_size)
            else:
                reduced_loss_avg = loss_avg.data
            torch.cuda.synchronize()
            losses_avg.update(reduced_loss_avg.item(), input.size(0))
            top1_avg.update(acc1_avg.item(), output.size(0))
            top5_avg.update(acc5_avg.item(), output.size(0))
    metrics = OrderedDict([
            ('top1', top1_avg.avg),
            ('top5', top5_avg.avg)])
    
    return metrics

def greedy_soup(population, score_lst, model, loader, args,amp_autocast = None):
    combined_lists = sorted(zip(score_lst, population), key=lambda x: x[0],reverse=True)
    score_lst, population = zip(*combined_lists)
    greedy_soup_ingredients = [population[0]]
    best_val_acc_so_far =test_model(model, loader, args, amp_autocast=amp_autocast)['top1']
    for i in range(1, len(population)):
        print("NO ",i)
        greedy_soup_ingredients_temp = greedy_soup_ingredients
        greedy_soup_ingredients_temp.append(population[i])
        greedy_soup_ingredients_temp_avg = torch.mean(torch.stack(greedy_soup_ingredients_temp), dim=0)
        model_weights_dict = model_vector_to_dict(model=model, weights_vector=greedy_soup_ingredients_temp_avg)
        model.load_state_dict(model_weights_dict)
        held_out_val_accuracy = test_model(model, loader, args, amp_autocast=amp_autocast)['top1']
        print("held_out_val_accuracy",held_out_val_accuracy)
        print("best_val_acc_so_far",best_val_acc_so_far)
        if held_out_val_accuracy > best_val_acc_so_far:
            print("held_out_val_accuracy",held_out_val_accuracy)
            greedy_soup_ingredients.append(population[i])
            best_val_acc_so_far = held_out_val_accuracy
            if args.local_rank == 0:
                print("The",i,"individual included")
    greedy_soup_ingredients_avg = torch.mean(torch.stack(greedy_soup_ingredients), dim=0)
    model_weights_dict = model_vector_to_dict(model=model, weights_vector=greedy_soup_ingredients_avg)
    if args.local_rank == 0:
        print("greedysoup result: ",best_val_acc_so_far)
    return model_weights_dict

def test_model_on_dataset(model, dataset,args):#test single model result in single
    model.eval()
    device = 'cuda'
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        end = time.time()
        loader = dataset.test_loader
        if type(dataset).__name__ == 'ImageNet2p':
            loader = dataset.train_loader
            # assert to make sure the imagenet held-out minival logic is consistent across machines.
            # tested on a few machines but if this fails for you please submit an issue and we will resolve.
            assert dataset.train_dataset.__getitem__(dataset.sampler.indices[1000])['image_paths'].endswith('n01675722_4108.JPEG')
        for i, batch in enumerate(loader):
            # if i > 1: break
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].cuda(), batch['labels'].cuda()
            data_time = time.time() - end
            y = labels
            if 'image_paths' in batch:
                image_paths = batch['image_paths']
            logits,output_list = model(input)
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)
            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            if isinstance(logits, list):
                logits = logits[0]
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths, None)
                correct += acc1
                n += num_total
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)
            functional.reset_net(model)
            batch_time = time.time() - end
            end = time.time()
            if i % 20 == 0 and args.local_rank == 0:
                percent_complete = 100.0 * i / len(loader)
                print(
                    f"[{percent_complete:.0f}% {i}/{len(loader)}]\t"
                    f"Acc: {100 * (correct/n):.2f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                )
        top1 = correct / n
        return top1
# def _transform(n_px):
#     return Compose([
#         Resize(n_px, interpolation=BICUBIC),
#         CenterCrop(n_px),
#         _convert_image_to_rgb,
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])

    
def test_ensemble_on_dataset(model, pop, popsize, dataset, args, amp_autocast=suppress): #test ensemble result in single
    population = pop
    model.eval()
    device = 'cuda'
    pop_avg = torch.mean(torch.stack(population), dim=0)
    # slice_len = args.slice_len or len(dataset)
    with torch.no_grad():
        top1, correct, n = 0., 0., 0. #ensemble
        top1_sm, correct_sm, n_sm = 0., 0., 0.
        # top1_mv, correct_mv, n_mv = 0., 0., 0.
        top1_avg, correct_avg, n_avg = 0., 0., 0.
        end = time.time()
        loader = dataset.test_loader
        if type(dataset).__name__ == 'ImageNet2p':
            loader = dataset.train_loader
            # assert to make sure the imagenet held-out minival logic is consistent across machines.
            # tested on a few machines but if this fails for you please submit an issue and we will resolve.
            assert dataset.train_dataset.__getitem__(dataset.sampler.indices[1000])['image_paths'].endswith('n01675722_4108.JPEG')
        for i,batch in enumerate(loader):
            # if i >10: break
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].cuda(), batch['labels'].cuda()
            pop_ouput = []
            pop_pred = []
            data_time = time.time() - end
            y = labels
            for j in range(popsize): #!!!
                solution = population[j]
                model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
                model.load_state_dict(model_weights_dict)
                model.eval()
                if 'image_paths' in batch:
                    image_paths = batch['image_paths']
                with amp_autocast():
                    logits,output_list = model(input)
                    # logits = model(inputs)
                projection_fn = getattr(dataset, 'project_logits', None)
                if projection_fn is not None:
                    logits = projection_fn(logits, device)
                if hasattr(dataset, 'project_labels'):
                    y = dataset.project_labels(y, device)
                if isinstance(logits, list):
                    logits = logits[0]
                functional.reset_net(model)
                
                pop_ouput.append(logits.unsqueeze(0))
                pred = logits.argmax(dim=1, keepdim=True).to(device)
                pop_pred.append(pred.unsqueeze(0))
            #mean of output layer ensemble
            output2 = torch.cat(pop_ouput, dim=0)
            output = torch.mean(output2, dim=0)
            _, pred2 = output.topk(1, 1, True, True)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(output, y, image_paths, None)
                correct += acc1
                n += num_total
            else:
                correct += pred.eq(y.view_as(pred2)).sum().item()
                n += y.size(0)
            #mean of softmax normlization and mean
            m = nn.Softmax(dim=2)
            output_sm = m(output2)
            output_sm = torch.mean(output_sm, dim=0)
            _, pred_sm = output_sm.topk(1, 1, True, True)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(output_sm, y, image_paths, None)
                correct_sm += acc1
                n_sm += num_total
            else:
                correct_sm += pred.eq(y.view_as(pred_sm)).sum().item()
                n_sm += y.size(0)
            # majority voting
            # outputt = torch.cat(pop_pred, dim=0)
            # acc1_mv = (outputt.mode(0).values == y).float().mean()*100.0

            #  weight Averaging
            model_weights_dict = model_vector_to_dict(model=model, weights_vector=pop_avg)
            model.load_state_dict(model_weights_dict)
            model.eval()
            with amp_autocast():
                output_avg,_= model(inputs)
            _, pred_avg = output_avg.topk(1, 1, True, True)
            functional.reset_net(model)
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits_avg = projection_fn(output_avg, device)
            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            if isinstance(logits, list):
                logits_avg = logits_avg[0]
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits_avg, y, image_paths, None)
                correct_avg += acc1
                n_avg += num_total
            else:
                correct_avg += pred.eq(y.view_as(pred_avg)).sum().item()
                n_avg += y.size(0)

            batch_time = time.time() - end
            end = time.time()
            if i % 20 == 0 and args.local_rank == 0:
                percent_complete = 100.0 * i / len(loader)
                print(
                    f"[{percent_complete:.0f}% {i}/{len(loader)}]\t"
                    f"Acc: {100 * (correct/n):.2f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                    f"Acc: {100 * (correct_sm / n_sm):.2f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                    f"Acc: {100 * (correct_avg / n_avg):.2f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                )
            # 
            top1 = correct / n
            top1_sm = correct_sm / n_sm
            # top1_mv = correct_mv / n_mv
            top1_avg= correct_avg / n_avg
        metrics = OrderedDict([('ensemble_top1',top1),
                ('wa_top1', top1_avg),
                ('ensemble_top1_sm', top1_sm),])
        
        return metrics

def test_ood(greedy_model_dict,args,model, population, popsize, amp_autocast=suppress):
    #test greedy_model and ensemble in different dataset
    results = {}
    ood_ensemble={}
    preprocess = transforms_imagenet_eval(img_size=224, crop_pct=0.875) #crop_pct=1
    accuracy_sum = 0
    for dataset_cls in [ImageNetV2, ImageNetR,ImageNetSketch,ObjectNet]: #ImageNetA 

        dataset = dataset_cls(preprocess, args.ood_path, args.batch_size, 4)
        model.load_state_dict(greedy_model_dict)
        accuracy = test_model_on_dataset(model, dataset,args)
        accuracy_sum += accuracy
        results[dataset_cls.__name__] = accuracy
        ood_ensemble[dataset_cls.__name__] = test_ensemble_on_dataset(model, population, popsize, dataset, args, amp_autocast=amp_autocast)
        if args.local_rank == 0:
            print(f'Evaluating on {dataset_cls.__name__} accuracy: {accuracy}')
    accuracy_avg = accuracy_sum/4
    metrics = OrderedDict([
            ('ImageNetV2_greedy',results['ImageNetV2']),
            ('ImageNetSketch_greedy',results['ImageNetSketch']),
            ('ImageNetR_greedy',results['ImageNetR']),
            ('ObjectNet_greedy',results['ObjectNet']),
            # ('ImageNetA_greedy',results['ImageNetA']),
            ('ood_avg_greedy',accuracy_avg),

            ('ImageNetV2_ensemble_ood_top1',ood_ensemble['ImageNetV2']['ensemble_top1']),
            ('ImageNetV2_wa_ood_top1',ood_ensemble['ImageNetV2']['wa_top1']),
            ('ImageNetV2_ensemble_top1_sm',ood_ensemble['ImageNetV2']['ensemble_top1_sm']),

            ('ImageNetSketch_ensemble_ood_top1',ood_ensemble['ImageNetSketch']['ensemble_top1']),
            ('ImageNetSketch_wa_ood_top1',ood_ensemble['ImageNetSketch']['wa_top1']),
            ('ImageNetSketch_ensemble_top1_sm',ood_ensemble['ImageNetSketch']['ensemble_top1_sm']),

            ('ImageNetR_ensemble_ood_top1',ood_ensemble['ImageNetR']['ensemble_top1']),
            ('ImageNetR_wa_ood_top1',ood_ensemble['ImageNetR']['wa_top1']),
            ('ImageNetR_ensemble_top1_sm',ood_ensemble['ImageNetR']['ensemble_top1_sm']),

            ('ObjectNet_ensemble_ood_top1',ood_ensemble['ObjectNet']['ensemble_top1']),
            ('ObjectNet_wa_ood_top1',ood_ensemble['ObjectNet']['wa_top1']),
            ('ObjectNet_ensemble_top1_sm',ood_ensemble['ObjectNet']['ensemble_top1_sm']),

            # ('ImageNetA_ensemble_ood_top1',ood_ensemble['ImageNetA']['ensemble_top1']),
            # ('ImageNetA_wa_ood_top1',ood_ensemble['ImageNetA']['wa_top1']),
            # ('ImageNetA_ensemble_top1_sm',ood_ensemble['ImageNetA']['ensemble_top1_sm']),

            ('ood_avg_ensemble_ood_top1',(ood_ensemble['ImageNetV2']['ensemble_top1']+ood_ensemble['ImageNetSketch']['ensemble_top1']+
                                          ood_ensemble['ImageNetR']['ensemble_top1']+ood_ensemble['ObjectNet']['ensemble_top1'])/4),
                                        #   ood_ensemble['ImageNetA']['ensemble_top1'])
            ('ood_avg_wa_ood_top1',(ood_ensemble['ImageNetV2']['wa_top1']+ood_ensemble['ImageNetSketch']['wa_top1']+
                                          ood_ensemble['ImageNetR']['wa_top1']+ood_ensemble['ObjectNet']['wa_top1'])/4),
                                        #   ood_ensemble['ImageNetA']['wa_top1'])/5),
            ('ood_avg_ensemble_top1_sm',(ood_ensemble['ImageNetV2']['ensemble_top1_sm']+ood_ensemble['ImageNetSketch']['ensemble_top1_sm']+
                                          ood_ensemble['ImageNetR']['ensemble_top1_sm']+ood_ensemble['ObjectNet']['ensemble_top1_sm'])/4),
                                        #   ood_ensemble['ImageNetA']['ensemble_top1_sm'])/5),
            ])
    return metrics  

def test_single_model_ood(args,model):
    results = {}
    acc_sum = 0
    preprocess = transforms_imagenet_eval(img_size=224, crop_pct=0.875) #crop_pct=1
    for dataset_cls in [ObjectNet,ImageNetV2, ImageNetR,ImageNetSketch]: #ImageNetA 
        dataset = dataset_cls(preprocess, args.ood_path, args.batch_size, 4)
        accuracy = test_model_on_dataset(model, dataset,args)
        results[dataset_cls.__name__] = accuracy
        acc_sum+=accuracy
        if args.local_rank == 0:
            print(f'Evaluating on {dataset_cls.__name__} accuracy: {accuracy}')
    acc_avg = acc_sum/4
    metrics = OrderedDict([
            ('ImageNetV2',results['ImageNetV2']),
            ('ImageNetSketch',results['ImageNetSketch']),
            ('ImageNetR',results['ImageNetR']),
            ('ObjectNet',results['ObjectNet']),
            # ('ImageNetA',results['ImageNetA']),
            ('ood_avg',acc_avg)
            ])
    return metrics  





