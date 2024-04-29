from random import random
from random import sample
from random import uniform
import numpy as np
import time
import torch
from timm.utils import *



def model_dict_to_vector(model):
    weights_vector = []
    for curr_weights in model.state_dict().values():
        vector = curr_weights.flatten().detach()
        weights_vector.append(vector)
    return torch.cat(weights_vector, 0)

def model_vector_to_dict(model, weights_vector):
    weights_dict = model.state_dict()
    start = 0
    for key in weights_dict:
        layer_weights_shape = weights_dict[key].shape
        layer_weights_size = weights_dict[key].numel()
        layer_weights_vector = weights_vector[start: start+layer_weights_size]
        weights_dict[key] = layer_weights_vector.view(layer_weights_shape).contiguous()
        start = start + layer_weights_size
    return weights_dict

# def model_weights_as_vector(model):
#     weights_vector = []

#     for curr_weights in model.state_dict().values():
#         curr_weights = curr_weights.detach().cpu().numpy()
#         vector = np.reshape(curr_weights, newshape=(curr_weights.size))
#         weights_vector.extend(vector)
#     return np.array(weights_vector)

# def model_weights_as_dict(model, weights_vector):
#     weights_dict = model.state_dict()
#     start = 0
#     for key in weights_dict:
#         w_matrix = weights_dict[key].detach().cpu().numpy()
#         layer_weights_shape = w_matrix.shape
#         layer_weights_size = w_matrix.size
#         layer_weights_vector = weights_vector[start:start + layer_weights_size]
#         layer_weights_matrix = np.reshape(layer_weights_vector, newshape=(layer_weights_shape))
#         weights_dict[key] = torch.from_numpy(layer_weights_matrix)
#         start = start + layer_weights_size
#     return weights_dict


def evolve4(cost_func,dim,popsize,population,population_f,paras,model, loader_de, args):
    update_label=[0 for i in range(popsize)]
    time1_m = AverageMeter()
    time2_m = AverageMeter()
    time3_m = AverageMeter()

    crm1, crm2, crm3, p1, p2, p3, p4, dyn_list_cr, dyn_list_nsf = paras
    pop = []
    list_probability = []
    list_cr1 = []
    list_cr2 = []
    list_cr3 = []
    device = population[0].device

    end = time.time()
    for idx in range(0, popsize):
        ## Calculate adaptive parameter cr and f
        cr1 = np.random.normal(crm1,  1e-9)
        cr1 = np.clip(cr1, 0, 0.2)
        list_cr1.append(cr1)
        
        cr2 = np.random.normal(crm2, 1e-9)
        cr2 = np.clip(cr2, 0, 0.2)
        list_cr2.append(cr2)
        
        cr3 = np.random.normal(crm3,  1e-9)
        cr3 = np.clip(cr3, 0, 0.2)
        list_cr3.append(cr3)


        # while True:
        #     f = np.random.normal(0.1, 0.04)
        #     if f < 0: continue
        #     elif f > 0.2: f = 0.2
        #     break
        f = np.random.normal(1e-5, 1e-8)

        # id1, id2, id3, id4, id5 = np.random.choice(list(set(range(0, popsize)) - {idx}), 5, replace=False)
        id1, id2, id3 = np.random.choice(list(set(range(0, popsize)) - {idx}), 3, replace=False)
        k = np.random.choice(4, 1, p=[p1,p2,p3,p4])
        print("f,cr",f,cr1,cr2,cr3)
        if k==0:
            x_new = population[id1] + f * (population[id2] - population[id3])
            pos_new = torch.where(torch.rand(dim).to(device) < torch.ones(dim).to(device)*cr1, x_new, population[idx])
            list_probability.append(k)
                             
        elif k==1:
            bestidx = population_f.index(min(population_f))
            x_new = population[idx] + f * (population[bestidx] - population[idx]) + \
                    f * (population[id2] - population[id3])
            pos_new = torch.where(torch.rand(dim).to(device) < torch.ones(dim).to(device)*cr2, x_new, population[idx])
            list_probability.append(k)
       
        # elif k==2:
        #     x_new = population[id1] + f * (population[id2] - population[id3]) + \
        #             f * (population[id4] - population[id5])
        #     pos_new = torch.where(torch.rand(dim).to(device) < torch.ones(dim).to(device)*cr3, x_new, population[idx])
        #     list_probability.append(k)
                             
        elif k==3 or k == 2:
            x_new = population[idx] + 0.2 * np.random.rand() * (population[id1] - population[idx]) + \
                    f * (population[id2] - population[id3])
            pos_new = x_new
            list_probability.append(k)  

        pop.append(pos_new)

    time1_m.update(time.time() - end) 
    end2 = time.time()
    # acc_list, new_population_f = cost_func(model, pop, loader_de, args)
    new_population_f = cost_func(model, pop, loader_de, args)
    time3_m.update(time.time() - end2) 

    for idx in range(0, popsize):
        t= new_population_f[idx]

        if list_probability[idx]==0:
            if t > (population_f[idx]):
                dyn_list_cr.append([list_cr1[idx],0,0])
                population[idx] = pop[idx]
                update_label[idx]=1
                population_f[idx] = t
                dyn_list_nsf.append([1,0,0,0,0,0,0,0])
            else:
                dyn_list_cr.append([-1,-1,-1])
                dyn_list_nsf.append([0,1,0,0,0,0,0,0])
                             
        elif list_probability[idx]==1:
            if t > (population_f[idx]):
                dyn_list_cr.append([0,list_cr2[idx],0])
                population[idx] = pop[idx]
                update_label[idx]=1
                population_f[idx] = t
                dyn_list_nsf.append([0,0,1,0,0,0,0,0])
            else:
                dyn_list_cr.append([-1,-1,-1])
                dyn_list_nsf.append([0,0,0,1,0,0,0,0])
                             
        elif list_probability[idx]==2:
            if t > (population_f[idx]):
                dyn_list_cr.append([0,0,list_cr3[idx]])
                population[idx] = pop[idx]
                update_label[idx]=1
                population_f[idx] = t
                dyn_list_nsf.append([0,0,0,0,1,0,0,0])
            else:
                dyn_list_cr.append([-1,-1,-1])
                dyn_list_nsf.append([0,0,0,0,0,1,0,0])
                             
        elif list_probability[idx]==3:
            if t > (population_f[idx]):
                population[idx] = pop[idx]
                update_label[idx]=1
                population_f[idx] = t
                dyn_list_cr.append([-1,-1,-1])
                dyn_list_nsf.append([0,0,0,0,0,0,1,0])
            else:
                dyn_list_cr.append([-1,-1,-1])
                dyn_list_nsf.append([0,0,0,0,0,0,0,1])

    dyn_list_cr_np = np.array(dyn_list_cr)
    dyn_list_nsf_np = np.array(dyn_list_nsf)

    last50 = dyn_list_cr_np[-20000:]
    # print("last50",last50)
    # print("dyn_list_cr_np",dyn_list_cr_np)
    if sum(last50[:,0]>0):
        crm1 = np.mean(last50[last50[:,0]>0,0])
    if sum(last50[:,1]>0):
        crm2 = np.mean(last50[last50[:,1]>0,1])
    if sum(last50[:,2]>0):
        crm3 = np.mean(last50[last50[:,2]>0,2])

    if dyn_list_cr_np.shape[0]>20000:
        dyn_list_cr = dyn_list_cr[-20000:]

    last50 = dyn_list_nsf_np[-20000:]
    # print("last50",last50)
    # print("dyn_list_nsf_np",dyn_list_nsf_np)
    k1 = k2 = k3 = k4 = 0.01
    t = np.sum(last50,0)
    # print("t",t)
    if t[0]+t[1] > 0 :
        k1 = max(t[0]/(t[0]+t[1]+0.01), 0.01)

    if t[2]+t[3] > 0 :
        k2 = max(t[2]/(t[2]+t[3]+0.01), 0.01)

    if t[4]+t[5] > 0 :
        k3 = max(t[4]/(t[4]+t[5]+0.01), 0.01)

    if t[6]+t[7] > 0 :
        k4 = max(t[6]/(t[6]+t[7]+0.01), 0.01)

    p1 = k1/(k1+k2+k3+k4)
    p2 = k2/(k1+k2+k3+k4)
    p3 = k3/(k1+k2+k3+k4)
    p4 = k4/(k1+k2+k3+k4)
    
    if dyn_list_nsf_np.shape[0]>20000:
        dyn_list_nsf = dyn_list_nsf[-20000:]
        
    bestidx = population_f.index(min(population_f))

    time2_m.update(time.time() - end) 
    print('time1_m: {time1.val:.3f} ({time1.avg:.3f})  '
          'time2_m: {time2.val:.3f} ({time2.avg:.3f})  '
          'time_score_func_m: {time3.val:.3f} ({time3.avg:.3f})  '.format(
            time1=time1_m, time2=time2_m, time3=time3_m))                 
    # print('p1, p2, p3, p4, f:', p1, p2, p3, p4, f)
    # population_f = new_population_f 
    return population, population_f, bestidx, [crm1, crm2, crm3, p1, p2, p3, p4, dyn_list_cr, dyn_list_nsf], update_label

if __name__ == '__main__':
    pass

    