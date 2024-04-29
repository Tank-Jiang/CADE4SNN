import os
from collections import OrderedDict

# population = [] 
# pop_init_dir = '/home/runhua/0814/resume/cifar100/output/train/20230816-001756-spikformer-32'
def get_path_with_acc(path):
    score = []
    acc1 = []
    acc5 = []
    val_loss = []
    models_path = []
    en_metrics = OrderedDict()
    with open(path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.split(', ')
        if 'ensemble' in line or 'wa' in line:
            en_metrics[line.split(':')[0].strip()] = float(line.split(':')[1].strip())
        elif len(parts) > 2:
            parts = line.split(', ')
            score.append(float(parts[0].split(':')[1].strip()))
            acc1.append(float(parts[1].split(':')[1].strip()))
            acc5.append(float(parts[2].split(':')[1].strip()))
            val_loss.append(float(parts[3].split(':')[1].strip()))
            models_path.append(parts[4].strip())
    return score, acc1, acc5, val_loss, en_metrics, models_path


def get_path(path):
    model_path_list = []
    with open(path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if len(line.split('/')) > 1:
            model_path_list.append(line.strip())
    return model_path_list

def write_path_with_acc(score, acc1, acc5, val_loss, en_metrics, models_path, pop_init):
    path = pop_init.split('.')[0]+'_score.txt'
    file = open(path, "w")
    for i in range(len(score)):
        line = 'score:' + str(score[i].item()) + ', acc1:' + str(acc1[i]) + ', acc5:' + str(acc5[i]) +  \
                                 ', val_loss:' + str(val_loss[i]) + ', ' + models_path[i] + '\n'
        file.writelines(line)

    file.writelines("ensemble_top1: " + str(en_metrics['ensemble_top1']) + '\n')
    file.writelines("ensemble_top5: " + str(en_metrics['ensemble_top5']) + '\n')
    file.writelines("ensemble_eval_loss: " + str(en_metrics['ensemble_eval_loss']) + '\n')
    file.writelines("wa_top1: " + str(en_metrics['wa_top1']) + '\n')
    file.writelines("wa_top5: " + str(en_metrics['wa_top5']) + '\n')
    file.writelines("wa_eval_loss: " + str(en_metrics['wa_eval_loss']) + '\n')
    file.writelines("ensemble_top1_sm: " + str(en_metrics['ensemble_top1_sm']) + '\n')
    file.writelines("ensemble_top5_sm: " + str(en_metrics['ensemble_top5_sm']) + '\n')
    file.writelines("ensemble_eval_loss_sm: " + str(en_metrics['ensemble_eval_loss_sm']) + '\n')
    file.close()
    return

# score, acc1, acc5, val_loss, en_metircs:
# [tensor(80.7324), tensor(80.8203)] 
# [73.25521087646484, 73.43750457763672] 
# [90.75521240234374, 90.72917175292969] 
# [1.473, 1.4738] 
# OrderedDict([('ensemble_top1', 73.33333587646484), ('ensemble_top5', 90.72917098999024), \
# ('ensemble_eval_loss', 3.9312), ('wa_top1', 73.41146087646484), \
# ('wa_top5', 90.70312957763672), ('wa_eval_loss', 1.4734), \
# ('ensemble_top1_sm', 73.33333587646484), ('ensemble_top5_sm', 90.72917098999024), \
# ('ensemble_eval_loss_sm', 0)])


