import time
from utils.tools.utility import *
import torch.nn as nn
from contextlib import suppress
from itertools import islice
from utils.tools.spe import model_vector_to_dict
from spikingjelly.clock_driven import functional

_logger = logging.getLogger('train')

def validate(model, loader, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    loss_fn = nn.CrossEntropyLoss().cuda()
    model.eval()
    end = time.time()
    slice_len = args.slice_len or len(loader)
    last_idx = slice_len - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(islice(loader, slice_len)):
            data_time_m.update(time.time() - end)
            last_batch = batch_idx == last_idx
            input, target = input.cuda(), target.cuda()
            input = input.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                output,_ = model(input)
            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data
            functional.reset_net(model)
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
    metrics = OrderedDict([('loss', round(losses_m.avg, 4)), 
                           ('top1', round(top1_m.avg,3)), ('top5', round(top5_m.avg,3))])
    if args.local_rank == 0: _logger.info('metrics_top1: {}'.format(metrics['top1']))
    return metrics


def validate_ensemble(model, pop, popsize, loader, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    losses_sm = AverageMeter()
    top1_sm = AverageMeter()
    top5_sm = AverageMeter()

    losses_avg = AverageMeter()
    top1_avg = AverageMeter()
    top5_avg = AverageMeter()

    population = pop
    end = time.time()
    loss_fn = nn.CrossEntropyLoss().cuda()
    pop_avg = torch.mean(torch.stack(population), dim=0)
    
    slice_len = args.slice_len or len(loader)
    last_idx = slice_len - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(islice(loader, slice_len)):
            # if batch_idx > 1: break
            data_time_m.update(time.time() - end)
            last_batch = batch_idx == last_idx
            input = input.cuda()
            target = target.cuda()
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
                    output,_ = model(input)
                functional.reset_net(model)
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
                output,_ = model(input)
            functional.reset_net(model)
            loss_avg = loss_fn(output, target)
            acc1_avg, acc5_avg = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)

                reduced_loss_sm = reduce_tensor(loss_sm.data, args.world_size)
                acc1_sm = reduce_tensor(acc1_sm, args.world_size)
                acc5_sm = reduce_tensor(acc5_sm, args.world_size)
                acc1_mv = reduce_tensor(acc1_mv, args.world_size)

                reduced_loss_avg = reduce_tensor(loss_avg.data, args.world_size)
                acc1_avg = reduce_tensor(acc1_avg, args.world_size)
                acc5_avg = reduce_tensor(acc5_avg, args.world_size)
            else:
                reduced_loss = loss.data
                reduced_loss_sm = loss_sm.data
                reduced_loss_avg = loss_avg.data
            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            losses_m.update(reduced_loss_sm.item(), input.size(0))
            top1_sm.update(acc1_sm.item(), output.size(0))
            top5_sm.update(acc5_sm.item(), output.size(0))

            # top1_m_mv.update(acc1_mv.item(), output.size(0))

            losses_avg.update(reduced_loss_avg.item(), input.size(0))
            top1_avg.update(acc1_avg.item(), output.size(0))
            top5_avg.update(acc5_avg.item(), output.size(0))

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

    metrics = OrderedDict([('ensemble_top1', top1_m.avg),
            ('ensemble_top5', top5_m.avg),
            ('ensemble_eval_loss', round(losses_m.avg, 4)),
            ('wa_top1', top1_avg.avg),
            ('wa_top5', top5_avg.avg),
            ('wa_eval_loss', round(losses_avg.avg, 4)),
            ('ensemble_top1_sm', top1_sm.avg),
            ('ensemble_top5_sm', top5_sm.avg),
            ('ensemble_eval_loss_sm', round(losses_sm.avg, 4))])
    if args.local_rank == 0: _logger.info('ensemble_metrics_top1: {}'.format(metrics['ensemble_top1']))
    
    return metrics


# out of distribution
# greedy