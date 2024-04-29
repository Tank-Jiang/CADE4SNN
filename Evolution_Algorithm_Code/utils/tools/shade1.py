from random import random
from random import sample
from random import uniform
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
# from timm.utils import *
from scipy.stats import cauchy
from collections import OrderedDict
from timm.utils import update_summary
import os
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

activation = 'identity'
if activation == 'tanh':
    act = torch.nn.Tanh()
else:
    act = torch.nn.Identity()

def regularization(pop_new, pop_old, afa=0, method='L2', associated=False):
    if associated == True:
        diff = pop_new - pop_old
        if method=='L2': 
            out = (diff**2).sum()
        elif method=='L1': 
            out = diff.abs().sum()
        else:
            out = torch.tensor(0)
    else:
        if method=='L2': 
            out = (pop_new**2).sum()-(pop_old**2).sum()
        elif method=='L1': 
            out = pop_new.abs().sum()-pop_old.abs().sum()
        else:
            out = torch.tensor(0)
    return afa * out.cpu()

# def evolve4(cost_func,epoch,current_generation,max_iters,population_init,bounds,dim,popsize,population,population_f,paras):
def evolve(score_func, epoch, de_iter, population, score_lst, paras1, paras2, model, loader_de, args):
    evolve_out = None
    if args.local_rank == 0:
        pop_new, params_for_update = mutate_and_crossover(epoch, de_iter, population, score_lst, paras1, paras2, args) #, *_ 
        pop_new_tensor = torch.stack(pop_new)
    # Distribute the tensor to all other available CUDA devices
    # if args.local_rank != 0:  pop_new = population                 
    if args.local_rank != 0:  pop_new_tensor = torch.stack(population) #[None for i in range(args.popsize)]
    # Synchronize all processes with a barrier
    if args.distributed: 
        dist.barrier()
        torch.distributed.broadcast(pop_new_tensor, src=0)
        dist.barrier() 

    pop_new = list(pop_new_tensor)

    # print('aaaaa', len(pop_new))
    # device = torch.device('cpu')
    # device = torch.cuda.current_device()
    # torch.cuda.set_device(device)
    # torch.distributed.broadcast_object_list(pop_new, src=0)
    # import pdb; pdb.set_trace() 

    new_population_f = score_func(model, pop_new, loader_de, args)

    if args.local_rank == 0:
        evolve_out = update_pop(epoch, de_iter, pop_new, population, new_population_f, score_lst, params_for_update, args)
    if args.distributed: 

        dist.barrier()
    return evolve_out
    # population, population_f, bestidx, worstidx, dist_matrics, paras2, update_label

def mutate_and_crossover(epoch, current_generation, population, population_f, paras1, paras2, args):
    # iters: iteration in each epoch
    # lp, cr_init, f_init, bounds, dim, popsize, max_iters, train_bool = paras1
    lp, cr_init, f_init, dim, popsize, max_iters, train_bool = paras1
    # time1_m = AverageMeter()
    # time2_m = AverageMeter()
    # time3_m = AverageMeter()
    # k_ls is the history index of memory for 4 mutation strategy
    # p1_c, p2_c, p3_c, p4_c : probability of using 4 strategies
    # ns_1, nf_1, ns_2, nf_2 : no of success/failure using both sin strategy
    # u_freq, u_f, u_cr: history memory of successful mean
    # k_ls: pointer to the table of history for each strategy
    p1_c, p2_c, p3_c, p4_c, p5_c, ns_1, nf_1, ns_2, nf_2, u_freq, u_f, u_cr, k_ls, dyn_list_nsf, succ_ls = paras2
    memory_size = u_f.shape[0] # size of history
    pop_new = []
    device = population[0].device
    # end = time.time()
    # 0 - trigonometrical 1 - sin all 2 - sin individual 3 - Guassian from SHADE
    strategy_ls = []
    # k_c_ls is for the choice of 4 strategies
    f_ls, cr_ls, freq_i_ls, k_c_ls = [], [], [], []
    for idx in range(0, popsize):
        freq_i = args.freq_init # for dummy in strategy other than 2
        freq = args.freq_init
        f = f_init
        cr = cr_init
        k_c = -1 # for dummy in strategy of 0 :trigonometrical
            
        ## An ensemble sinusoidal parameter adaptation incorporated with L-SHADE for solving CEC2014 benchmark problems
        ## L-SHADE-Epsin
        ## DOI:10.1109/CEC.2016.7744163.
        ## Sepearate to two half of generations

        ## define f: scaling factor
        k_c = np.random.choice(5, 1, p=[p1_c, p2_c, p3_c, p4_c, p5_c]).item()
        # id1, id2, id3, id4, id5 = np.random.choice(list(set(range(0, popsize)) - {idx}), 5, replace=False)
        id1, id2, id3, id4 = np.random.choice(list(set(range(0, popsize)) - {idx}), 4, replace=False)
        if k_c<4:
            if current_generation < max_iters/2:
                # k_c = np.random.choice(4, 1, p=[p1_c,p2_c,p3_c,p4_c]).item()
    #            if current_generation <= lp:
                if epoch == 0 and current_generation <= lp:
                    # 2.a Both sinusoidal configurations have the same probability
                    p1 = 0.5
                    p2 = 0.5
                else:
                    success_option_1 = np.sum(ns_1) / (np.sum(ns_1) + np.sum(nf_1)) + 0.01
                    success_option_2 = np.sum(ns_2) / (np.sum(ns_2) + np.sum(nf_2)) + 0.01

                    p1 = success_option_1 / (success_option_1 + success_option_2)
                    p2 = success_option_2 / (success_option_1 + success_option_2)

                if p1 > np.random.rand():
                    f = f_init * (np.sin(2*np.pi*freq*current_generation + np.pi)*(max_iters-current_generation)/max_iters + 1)
                    strategy_ls.append(1)
                    # print('freq:', freq, 'f:', f)
                    # pdb.set_trace()
                else:
                    random_index = torch.randint(0, memory_size, size=(1,)).item()
                    # freq_i = torch.normal(torch.tensor(u_freq[random_index, k_c]), 0.1)
                    freq_i = np.random.normal(u_freq[random_index, k_c].item(), args.freq_std)
                    # freq_i = cauchy.rvs(loc=u_freq[random_index, k_c].item(), scale=0.1, size=1)[0]
                    f = f_init * (np.sin(2*np.pi*freq_i*current_generation)*current_generation/max_iters + 1)
                    strategy_ls.append(2)
                    # print('freq_i:', freq_i, 'f:', f)
                # pdb.set_trace()
            else:
                random_index = np.random.randint(0, memory_size)
                # f = torch.normal(torch.tensor(u_f[random_index, k_c]), 0.1)
                # f = np.random.normal(u_f[random_index, k_c], 0.01)
                # f = np.clip(f, 0.0001, 0.3)
                f = np.random.normal(u_f[random_index, k_c], args.f_std)
                # f = cauchy.rvs(loc=u_f[random_index, k_c].item(), scale=0.1, size=1)[0]
                f = np.clip(f, 0.0001, args.f_clip)
                strategy_ls.append(3)
                # pdb.set_trace()

            # Crossover
            random_index = np.random.randint(0, memory_size)
            # cr = torch.normal(torch.tensor(u_cr[random_index, k_c]), 0.1)
            cr = np.random.normal(u_cr[random_index, k_c].item(), args.cr_std)
            cr = np.clip(cr, 0, args.cr_clip)
            # if u_cr[random_index,k_c] == 0.4 or cr == 0.4: cr = 0.2
            # pdb.set_trace()
            # f = torch.tensor(f, dtype=torch.float32)
        bestidx = population_f.index(min(population_f))
        if k_c==0:
            x_new = population[idx] + f * (population[id1] - population[idx])
            pos_new = torch.where(torch.rand(dim).to(device) < torch.ones(dim).to(device)*cr, x_new, population[idx])

        elif k_c==1:
            x_new = population[idx] + f * (population[bestidx] - population[idx])
            pos_new = torch.where(torch.rand(dim).to(device) < torch.ones(dim).to(device)*cr, x_new, population[idx])

        elif k_c==2:
            x_new = population[idx] + f * (population[id1] - population[idx]) + \
                    f * (population[bestidx] - population[idx])
            pos_new = torch.where(torch.rand(dim).to(device) < torch.ones(dim).to(device)*cr, x_new, population[idx])

        elif k_c==3:
            x_new = population[idx] + f * np.random.rand() * (population[bestidx] - population[idx]) + \
                    f * (population[id1] - population[idx])
            pos_new = torch.where(torch.rand(dim).to(device) < torch.ones(dim).to(device)*cr, x_new, population[idx])

        elif k_c==4:
            ## Trigonometric Mutation Operation Journal of Global Optimization 27: 105–129, 2003.
            id1, id2, id3 = np.random.choice(list(set(range(0, popsize)) - {idx}), 3, replace=False)
            p_total = population_f[id1] + population_f[id2] + population_f[id3]
            p1_, p2_, p3_ =  population_f[id1]/p_total, population_f[id2]/p_total, population_f[id3]/p_total
            x_new = population[id1]*(1./3 + p2_ + p3_ - 2*p1_) + population[id2]*(1./3 + p1_ + p3_ - 2*p2_) +\
                    population[id3]*(1./3 + p1_ + p2_ - 2*p3_)
            pos_new = x_new
            strategy_ls.append(0)

        pos_new = torch.where(train_bool, pos_new, population[idx])
        # print(current_generation, 'u_f:', u_f, 'u_cr:', u_cr)
        f_ls.append(f)
        cr_ls.append(cr)
        freq_i_ls.append(freq_i)
        k_c_ls.append(k_c)
        pop_new.append(pos_new)

    params_for_update = [memory_size, lp, max_iters, strategy_ls, ns_1, nf_1, ns_2, nf_2, u_freq, u_f, u_cr, \
                                                    k_ls, f_ls, cr_ls, freq_i_ls, k_c_ls, dyn_list_nsf, succ_ls, train_bool]
    return pop_new, params_for_update
    # time1_m.update(time.time() - end)
    # end2 = time.time()
    # print('new_population_f:........', new_population_f)
    # new_population_f = cost_func(pop)
    # time3_m.update(time.time() - end2)
def update_pop(epoch, current_generation, pop_new, population, new_population_f, population_f, params_for_update, args, **kwargs):
    memory_size, lp, max_iters, strategy_ls, ns_1, nf_1, ns_2, nf_2, u_freq, u_f, u_cr, \
                                k_ls, f_ls, cr_ls, freq_i_ls, k_c_ls, dyn_list_nsf, succ_ls, train_bool = params_for_update
    popsize = len(population)
    update_label=[0 for i in range(popsize)]
    regul = [regularization(pop_new[i], population[i]) for i in range(popsize)]
    winners = np.array(new_population_f)+np.array(regul) < np.array(population_f) 


    # output to csv file to record the successful/unsuccessful trials
    for idx in range(0, popsize): 
        de_iter_dict = OrderedDict([('iter', current_generation)])
        parameter3=OrderedDict([('strategy',strategy_ls[idx]), ('k_c_ls',k_c_ls[idx]), ('freq', freq_i_ls[idx]), ('f', f_ls[idx]), 
            ('cr', cr_ls[idx]), ('succ',winners[idx])])
        update_summary(epoch, de_iter_dict, parameter3, os.path.join(args.output_dir, 'summary_strategy.csv'), write_header=True) 



    if epoch % 5==0 and current_generation==1:
        succ_ls[2,:] = 0
        succ_ls[3,:] = 0
    # update successful list
    for idx in range(0, popsize):
        if strategy_ls[idx]==0:
            succ_ls[0,0] += 1
            succ_ls[2,0] += 1
            if winners[idx]: 
                succ_ls[1,0] += 1
                succ_ls[3,0] += 1
        else: 
            # start from 1
            succ_ls[0, strategy_ls[idx]*4+k_c_ls[idx] -3] += 1
            succ_ls[2, strategy_ls[idx]*4+k_c_ls[idx] -3] += 1
            if winners[idx]: 
                succ_ls[1,strategy_ls[idx]*4+k_c_ls[idx] -3] += 1
                succ_ls[3,strategy_ls[idx]*4+k_c_ls[idx] -3] += 1

    # Update success lists to recalculate probabilities
    if current_generation <= (max_iters / 2):
        if len(ns_1) == lp:  del ns_1[0], ns_2[0], nf_1[0], nf_2[0]

        ns_1.append(np.sum(winners * (np.array(strategy_ls) == 1)))
        ns_2.append(np.sum(winners * (np.array(strategy_ls) == 2)))
        nf_1.append(np.sum(~winners * (np.array(strategy_ls) == 1)))
        nf_2.append(np.sum(~winners * (np.array(strategy_ls) == 2)))
    ## Update memory
    ## only count the sucessful one
    ## exclude the trigonometrical one
    for ik in range(4): ## four update variants
        indexes = np.where(winners * (np.array(k_c_ls) == ik))
        print('winners:.....', winners, 'k_c_ls:.....', k_c_ls)
        if np.sum(indexes) > 0:
            weights = np.abs(np.array(population_f)[indexes] - np.array(new_population_f)[indexes])
            weights /= np.sum(weights)

            u_cr[k_ls[ik],ik] = np.sum(weights * np.array(cr_ls)[indexes] ** 2) / np.sum(weights * np.array(cr_ls)[indexes])
            u_f[k_ls[ik],ik] = np.sum(weights * np.array(f_ls)[indexes] ** 2) / np.sum(weights * np.array(f_ls)[indexes])

            if  current_generation < (max_iters / 2):
                chosen = (np.array(strategy_ls) == 2) * winners
                chosen = np.where(chosen * (np.array(k_c_ls) == ik))
                weights2 = np.abs(np.array(population_f)[chosen] - np.array(new_population_f)[chosen])
                weights2 /= np.sum(weights2)
                if np.sum(chosen) != 0:   
                    u_freq[k_ls[ik],ik] = np.sum(weights2 * np.array(freq_i_ls)[chosen] ** 2) / np.sum(weights2 * np.array(freq_i_ls)[chosen])
                    #u_freq[k_ls[ik],ik] = np.mean(np.array(freq_i_ls)[chosen])
                if np.isnan(u_freq[k_ls[ik],ik]):   u_freq[k_ls[ik],ik] = 0.5
            else:
                u_freq[k_ls[ik],ik] = freq_i_ls[0]
            k_ls[ik] += 1
            if k_ls[ik] == memory_size: k_ls[ik] = 0
    
    for idx in range(0, popsize):
        t= new_population_f[idx]
        t2 = t + regul[idx]
        # for strategy 1-3, exclude the trigonometrical one which give kc=-1
        if k_c_ls[idx]==0:
            if t2 < (population_f[idx]):
                dyn_list_nsf.append([1,0,0,0,0,0,0,0,0,0])
            else:
                dyn_list_nsf.append([0,1,0,0,0,0,0,0,0,0])

        elif k_c_ls[idx]==1:
            if t2 < (population_f[idx]):
                dyn_list_nsf.append([0,0,1,0,0,0,0,0,0,0])
            else:
                dyn_list_nsf.append([0,0,0,1,0,0,0,0,0,0])

        elif k_c_ls[idx]==2:
            if t2 < (population_f[idx]):
                dyn_list_nsf.append([0,0,0,0,1,0,0,0,0,0])
            else:
                dyn_list_nsf.append([0,0,0,0,0,1,0,0,0,0])

        elif k_c_ls[idx]==3:
            if t2 < (population_f[idx]):
                dyn_list_nsf.append([0,0,0,0,0,0,1,0,0,0])
            else:
                dyn_list_nsf.append([0,0,0,0,0,0,0,1,0,0])
        elif k_c_ls[idx]==4:
            if t2 < (population_f[idx]):
                dyn_list_nsf.append([0,0,0,0,0,0,0,0,1,0])
            else:
                dyn_list_nsf.append([0,0,0,0,0,0,0,0,0,1])
        if t2 <= (population_f[idx]):
            population[idx] = pop_new[idx]
            update_label[idx]=1
            population_f[idx] = t

    dyn_list_nsf_np = np.array(dyn_list_nsf)

    # number of trials in recent 5 epoch
    recent_size = 5*max_iters*popsize
    ## for choosing strategy 1-4
    if len(dyn_list_nsf_np)>0:
        last50 = dyn_list_nsf_np[-recent_size:]
    else:
        last50 = [[0,0,0,0,0,0,0,0,0,0]]

    k1 = k2 = k3 = k4 = k5 = 0.01
    t = np.sum(last50,0)


    de_iter_dict = OrderedDict([('iter', current_generation)])
    parameter3=OrderedDict([('counts',t)])
    update_summary(epoch, de_iter_dict, parameter3, os.path.join(args.output_dir, 'summary_strategy2.csv'), write_header=True) 

    if t[0]+t[1] > 0 : k1 = max((t[0]+0.01)/(t[0]+t[1]+0.01), 0.01)
    if t[2]+t[3] > 0 : k2 = max((t[2]+0.01)/(t[2]+t[3]+0.01), 0.01)
    if t[4]+t[5] > 0 : k3 = max((t[4]+0.01)/(t[4]+t[5]+0.01), 0.01)
    if t[6]+t[7] > 0 : k4 = max((t[6]+0.01)/(t[6]+t[7]+0.01), 0.01)
    if t[8]+t[9] > 0 : k5 = max((t[8]+0.01)/(t[8]+t[9]+0.01), 0.01)

    # if t[0]+t[1] > 0 : k1 = max(t[0]/(t[0]+t[1]+0.01), 0.01)
    # if t[2]+t[3] > 0 : k2 = max(t[2]/(t[2]+t[3]+0.01), 0.01)
    # if t[4]+t[5] > 0 : k3 = max(t[4]/(t[4]+t[5]+0.01), 0.01)
    # if t[6]+t[7] > 0 : k4 = max(t[6]/(t[6]+t[7]+0.01), 0.01)
    # if t[8]+t[9] > 0 : k5 = max(t[8]/(t[8]+t[9]+0.01), 0.01)

    ktot = k1+k2+k3+k4+k5
    p1_c, p2_c, p3_c, p4_c, p5_c = k1/ktot, k2/ktot, k3/ktot, k4/ktot, k5/ktot
    if p5_c>0.8:
        ktot = k1+k2+k3+k4
        p1_c, p2_c, p3_c, p4_c, p5_c = k1/ktot*0.2, k2/ktot*0.2, k3/ktot*0.2, k4/ktot*0.2, 0.8
    
    if dyn_list_nsf_np.shape[0]>recent_size: dyn_list_nsf = dyn_list_nsf[-recent_size:]

    bestidx = population_f.index(min(population_f))
    worstidx = population_f.index(max(population_f))
    population_matric = torch.stack(population)
    # population_init_matric = torch.stack(population_init)

    # exclude the BN layer 
    population_matric = population_matric*train_bool
    # population_init_matric = population_init_matric*train_bool


    # cons_sim = F.cosine_similarity(population_matric, population_init_matric)
    # cons_sim = cons_sim.abs().mean()
    cons_sim = 0
    # l2_dist = F.pairwise_distance(population_matric, population_init_matric, eps=1e-6, keepdim=False).mean()
    l2_dist = 0
    eucl_dist = F.pdist(population_matric, p=2)
    lowest_dist, mean_dist, largest_dist = eucl_dist.min(), eucl_dist.mean(), eucl_dist.max()
    L1_value = torch.mean(population_matric.abs(), dim=1)

    L2_value =  torch.mean(population_matric**2, dim=1)
    dist_matrics = [cons_sim, l2_dist, lowest_dist, mean_dist, largest_dist,L1_value,L2_value]

    paras2 = [p1_c, p2_c, p3_c, p4_c, p5_c, ns_1, nf_1, ns_2, nf_2, u_freq, u_f, u_cr, k_ls, dyn_list_nsf, succ_ls]
    # time2_m.update(time.time() - end)
    print('update_label........:',update_label,p1_c, p2_c, p3_c, p4_c, p5_c)
    # print('time1_m: {time1.val:.3f} ({time1.avg:.3f})  '
    #       'time2_m: {time2.val:.3f} ({time2.avg:.3f})  '
    #       'time_score_func_m: {time3.val:.3f} ({time3.avg:.3f})  '.format(
    #         time1=time1_m, time2=time2_m, time3=time3_m))

    return population, population_f, bestidx, worstidx, dist_matrics, paras2, update_label

if __name__ == '__main__':
    import pdb
    # pdb.set_trace()
    memory_size = 5
    model,loader,ckp = 0, 0, 0
    popsize = 10
    dim = 20
    population = torch.randn(popsize, dim).tolist()
    # population_init = torch.randn(popsize, dim).tolist()
    # population = torch.randn(popsize, dim).cuda().tolist()
    population = [torch.Tensor(i) for i in population]
    # population_init = [torch.Tensor(i) for i in population_init]
    # population_matric = torch.stack(population, dim=0)
    bounds = 1
# ***************************************
    # need to initialize in the main
    k_ls = [0,0,0,0]
    cr_init = 0.2
    f_init = 0.05
    #  u_freq, u_f, u_cr with size (memory_size, 4)
    # 1.2 Initialize memory of first control settings
    u_f = np.ones((memory_size,4)) * f_init
    u_cr = np.ones((memory_size,4)) * cr_init
    # 1.3 Initialize memory of second control settings
    u_freq = np.ones((memory_size,4)) * .5
    ns_1 = []
    nf_1 = []
    ns_2 = [] 
    nf_2 = []
    dyn_list_nsf = []
# ***************************************
    p1_c, p2_c, p3_c, p4_c, p5_c = 0.2,0.2,0.2,0.2,0.2
    paras = [p1_c, p2_c, p3_c, p4_c, p5_c, ns_1, nf_1, ns_2, nf_2, u_freq, u_f, u_cr, k_ls, dyn_list_nsf] 

    def cost_func(population):
        population_f = [((i+1)/2).sum() for i in population]
        return population_f

    for epoch in range(2):
        max_iters = 50
        population_f = cost_func(population)
        for current_generation in range(max_iters):
            population, population_f, bestidx,worstidx, dist_matrics, paras, update_label = evolve(cost_func,epoch,current_generation, \
                                                                        max_iters,bounds,dim, \
                                                                        popsize,population,population_f, \
                                                                        paras)
            print('epoch:', epoch, 'current_generation:', current_generation, 'bestidx:', bestidx,'worstidx',worstidx)
            [p1_c, p2_c, p3_c, p4_c, p5_c, ns_1, nf_1, ns_2, nf_2, u_freq, u_f, u_cr, k_ls, dyn_list_nsf] = paras
            cons_sim, l2_dist, lowest_dist, mean_dist, largest_dist = dist_matrics
    # print('update_label:', update_label, '\n')
    print('p1_c, p2_c, p3_c, p4_c:', p1_c, p2_c, p3_c, p4_c, '\n')
    print('ns_1, nf_1, ns_2, nf_2: \n', ns_1, nf_1, ns_2, nf_2, '\n')
    print('u_freq, u_f, u_cr, k_ls: \n', u_freq, u_f, u_cr, k_ls, '\n')
    print('length of dyn_list_nsf:', len(dyn_list_nsf), '\n')
    print('cons_sim, l2_dist, lowest_dist, mean_dist, largest_dist:', cons_sim, l2_dist, lowest_dist, mean_dist, largest_dist, '\n')
    pdb.set_trace()

# import torch
# population_init = torch.randn(100, 128)
# population = torch.randn(100, 128)
# cons_sim = F.cosine_similarity(population_init, population)
# # print(cons_sim.shape)
# cons_sim = cons_sim.abs().mean()
# l2_dist = F.pairwise_distance(population_init, population, eps=1e-6, keepdim=False).mean()
# eucl_dist = F.pdist(population, p=2)
# lowest_dist, mean_dist, largest_dist = eucl_dist.min(), eucl_dist.mean(), eucl_dist.max()


        
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

#***************************************
## need to initialize in the main
##  k_ls = [0,0,0,0]
##  u_freq, u_f, u_cr with size (memory_size, 4)
#    # 1.2 Initialize memory of first control settings
#    u_f = np.ones(memory_size,4) * .5
#    u_cr = np.ones(memory_size,4) * .5
#    # 1.3 Initialize memory of second control settings
#    u_freq = np.ones(memory_size,4) * .5
#    ns_1 = []
#    nf_1 = []
#    ns_2 = [] 
#    nf_2 = []
#***************************************

# #####Gausian Walk
    
# ## Gaussian Walk based on best result
# ## An ensemble sinusoidal parameter adaptation incorporated with L-SHADE for solving
# ## CEC2014 benchmark problems. July 2016. DOI:10.1109/CEC.2016.7744163.
# ## y_i =Gaussian(μb,σ)+(e1*x_best −e2*x_i)
# if current_generation == max_iters and epoch > 1000:
# # if current_generation == 1:
#     for i_gen in range(100, 201, 50):
#     # for i_gen in range(2, 250):
#         success_num = 0
#         new_population = []
#         ## prepare the child by Guassian walk around best solution
#         bestidx = population_f.index(min(population_f))
#         worstidx = population_f.index(max(population_f))
#         for idx in range(0, popsize//2):
#             sigma = torch.abs(np.log(i_gen)/i_gen*(population[idx] - population[bestidx]))
#             # pdb.set_trace()
#             # a = torch.normal(0,1, (10,))； b = torch.normal(a, 1) - a #b的正负号是随机的
#             # pos_new = torch.normal(population[bestidx], sigma) + \
#             #           + np.random.rand()*population[bestidx] - np.random.rand()*population[idx]
#             pos_new = torch.normal(population[bestidx], sigma) + \
#                       + 0.1*np.random.rand()*(population[bestidx] - population[idx])
#             new_population.append(pos_new)
#         new_population_f,new_population_f_without_L2 = cost_func(new_population)
#         # new_population_f = [i+np.random.normal() for i in population_f[:popsize//2]]
        
#         sorted_population_f, old_idx = torch.sort(torch.tensor(population_f), descending=True)
#         sorted_new_population_f, new_idx = torch.sort(torch.tensor(new_population_f), descending=False)
#         for i in old_idx[:popsize//2]:
#             if new_population_f[0] < population_f[i]: 
#                 update_label[i] = 1
#                 success_num += 1
#                 population_f[i] = new_population_f[0]
#                 population[i] = new_population[0]
#                 del new_population_f[0]
#                 del new_population[0]
#         print('epoch:', epoch, 'current_generation:', current_generation, 'i_gen:', i_gen, 'success_num!!!!!', success_num)

#         # ## arrange list of child
#         # childidx = torch.topk(pop, popsize//2, largest=False, sorted=True)
#         # ## check worst parent
#         # worstidx = torch.topk(population_f, popsize//2, largest=True, sorted=True)
#         # newidx, oldidx = 0, 0  ## sort index of both lists
#         # while newidx < popsize//2 and oldidx < popsize//2:
#         #     ## replacement happens
#         #     if new_population_f[childidx[newidx]] < population_f[worstidx[oldidx]]:
#         #         population[worstidx[oldidx]] = pop[childidx[newidx]]
#         #         population_f[worstidx[oldidx]] = new_population_f[childidx[newidx]]
#         #         newidx += 1
#         #         oldidx += 1
#         #     ## keep the parent
#         #     else:
#         #         oldidx += 1
# # pdb.set_trace()













