import pandas as pd
import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt
# import wandb
from matplotlib.lines import Line2D
import os
import ast
def plot_top1_vs_baseline(readpath,savepath, exp_name,wandb=None):
    readpath = os.path.join(readpath,'summary.csv')
    savepath = os.path.join(savepath, 'plot_top1_vs_baseline_'+exp_name+'.svg')
    
    df = pd.read_csv(readpath, skipinitialspace=True)
    ensemble_top1 = float(df.iloc[0, 1])
    wa_top1 = float(df.iloc[0, 4])
    ensemble_top1_sm = float(df.iloc[0, 7])
    baseline_values = [ensemble_top1, wa_top1, ensemble_top1_sm]
    init_pop_top1 = eval(df.iloc[2,2])
    top1_values_str = [x for x in df.iloc[3:, 3].dropna().tolist() if x != 'top1']
    top1_values = [ast.literal_eval(x) for x in top1_values_str]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot the horizontal lines
    ax.axhline(y=ensemble_top1, color='r', linestyle='--', label='ensemble')
    ax.axhline(y=wa_top1, color='g', linestyle='--', label='weight averaging')
    ax.axhline(y=76.04, color='purple', linestyle='--', label='Initialization') 
    
    # Add markers at the start of the baselines
    ax.scatter(0, ensemble_top1, marker='o', facecolors='none', edgecolors='r', s=30)
    ax.scatter(0, wa_top1, marker='s', facecolors='none', edgecolors='g', s=30)
    ax.scatter(0, 76.04, marker='s', facecolors='none', edgecolors='purple', s=30)
    
    # print(top1_values)
    # print(len(top1_values))
    ax.plot([0] * len(init_pop_top1), init_pop_top1, 'x', color='red', alpha=0.5, label='init pop')

    for epoch, values in enumerate(top1_values):
        # if epoch == 0:
        #     ax.scatter([epoch] * len(values), values, marker='D', facecolors='none', edgecolors='blue', alpha=0.5, s=15, label='finetuned')
        # else:
        #     ax.plot([epoch] * len(values), values, 'x', color='blue', alpha=0.5, label='finetuned' if epoch == 1 else "")
        ax.plot([epoch+1] * len(values), values, 'x', color='blue', alpha=0.5, label='finetuned' if epoch == 1 else "")
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markeredgecolor='r', label='greedy soup'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='none', markeredgecolor='g', label='weight averaging'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='none', markeredgecolor='purple', label='Initialization'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor='none', markeredgecolor='blue', label='finetuned')
    ]
    ax.set_title('Top1 vs Baselines')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Top1 Accuracy')
    ax.legend(handles=legend_elements)
    ax.grid(True)
    
    # ax.set_xlim(-0.2, len(top1_values) - 0.5)
    
    min_baseline = min(baseline_values)
    max_value = max([max(v) for v in top1_values])
    # ax.set_ylim(min_baseline - 0.5, max_value + 0.2)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    plt.savefig(savepath)
    
    if wandb is not None:
        wandb.log({'Top1_vs_Baselines': wandb.Image(fig)})
        
    plt.show()
# plot_top1_vs_baseline('/home/guodong/runhua/spe_final_version_ann/output/train/20230921-171618-resnet50_2nd',
#                       '/home/guodong/runhua/spe_final_version_ann_good_result/result',"2nd",None)
# Usage
def plot_loss(output_dir, pop_size, wandb=None):
    readpath, savepath = os.path.join(output_dir,'summary.csv'), os.path.join(output_dir, 'plot.svg') 
    df = pd.read_csv(readpath)
    val_acc=[]
    val_loss=[]
    bestidx=[]
    worstidx=[]
    for i in range(len(df["eval_top1"])):
        if(i%2==0):
            val_acc.append(eval(df["eval_top1"][i]))
            val_loss.append(eval(df["eval_eval_loss"][i]))
            bestidx.append(int(df['train_bestidx'][i]))
            worstidx.append(int(df['train_worstidx'][i]))
    
    val_acc=np.array(val_acc)
    val_loss=np.array(val_loss)
    bestidx=bestidx[1:]
    worstidx=worstidx[1:]
    new_pop_acc = []
    old_pop_acc=[]
    new_pop_loss = []
    old_pop_loss=[]
    for i in range(len(val_acc)):
        for j in range(len(val_acc[0])):
            if i == 0:
                old_pop_acc.append([i,val_acc[i][j]])
                old_pop_loss.append([i,val_loss[i][j]])
            else:
                new_pop_acc.append([i,val_acc[i][j]])
                new_pop_loss.append([i,val_loss[i][j]])
    old_pop_acc=np.array(old_pop_acc)
    new_pop_acc=np.array(new_pop_acc)

    # val_loss=[]
    # for i in range(len(df["eval_eval_loss"])):
    #     if(i%2==0):
    #         val_loss.append(eval(df["eval_eval_loss"][i]))

    # val_loss=np.array(val_loss)
    # new_pop_loss = []
    # old_pop_loss=[]
    # for i in range(len(val_loss)):
    #     for j in range(len(val_loss[0])):
    #         if i == 0:
    #             old_pop_loss.append([i,val_loss[i][j]])
    #         else:
    #             new_pop_loss.append([i,val_loss[i][j]])
    old_pop_loss=np.array(old_pop_loss)
    new_pop_loss=np.array(new_pop_loss)
    adam_acc=val_acc[0]
    adam_loss=val_loss[0]   

#____plot the best individual for every population 

    bestidx_acc = new_pop_acc[bestidx + np.arange(0, len(bestidx) * pop_size, pop_size)]
    bestidx_loss = new_pop_loss[bestidx + np.arange(0, len(bestidx) * pop_size, pop_size)]
    worstidx_acc = new_pop_acc[worstidx + np.arange(0, len(worstidx) * pop_size, pop_size)]
    worstidx_loss = new_pop_loss[worstidx + np.arange(0, len(worstidx) * pop_size, pop_size)]
    optimizer_change_point=pop_size
    scaling1=len(new_pop_acc)*2/(pop_size*3*pop_size)
    optimizer_change_point=pop_size
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    
    ax.plot([x*scaling1 if x<=0 else x for x in range(-optimizer_change_point,0)],adam_loss[:optimizer_change_point],color='#0557FA',alpha=1)
    ax.scatter(x=new_pop_loss[:,0],y=new_pop_loss[:,1],marker='x',color='#F35858',)
    ax.scatter(x=old_pop_loss[:,0],y=old_pop_loss[:,1],marker='x',color='#007FFF',)
    ax.scatter(bestidx_loss[:,0],bestidx_loss[:,1],marker='o',color='#33FF99')
    ax.plot(bestidx_loss[:,0],bestidx_loss[:,1],color='#33FF99',alpha=1)

    ax.scatter(worstidx_loss[:,0],worstidx_loss[:,1],marker='o',color='#ffddb7')
    ax.plot(worstidx_loss[:,0],worstidx_loss[:,1],color='#ffddb7',alpha=1)
    # ax.scatter(x=pop_loss_best[:,0],y=pop_loss_best[:,1],marker='o',color='#33FF99')#!!!!
    
    # ax.plot(pop_loss_best[:,0],pop_loss_best[:,1],color='#33FF99',alpha=1)#!!!!
    
    ax.tick_params(labelsize = 20)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(pad = 10)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    plt.yticks(fontsize = 20)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    for tick in ax.xaxis.get_ticklines():
        tick.set_markersize(3)
        tick.set_markeredgewidth(0.5)
    for tick in ax.yaxis.get_ticklines():
        tick.set_markersize(3)
        tick.set_markeredgewidth(0.5)
    left, bottom, width, height = [0.548, 0.53, 0.328, 0.34]
    ax1 = fig.add_axes([left, bottom, width, height])

    ax1.plot([x*scaling1 if x<=0 else x for x in range(-optimizer_change_point,0)],adam_acc[:optimizer_change_point],color='#0557FA',alpha=1)
    ax1.scatter(x=new_pop_acc[:,0],y=new_pop_acc[:,1],marker='x',color='#F35858')
    ax1.scatter(x=old_pop_acc[:,0],y=old_pop_acc[:,1],marker='x',color='#007FFF')
#____plot the best individual for every population 
    # ax1.scatter(pop_acc_best[:,0],pop_acc_best[:,1],marker='o',color='#33FF99')#!!!!    
    # ax1.plot(pop_acc_best[:,0],pop_acc_best[:,1],color='#33FF99',alpha=1)#!!!!
    ax1.scatter(bestidx_acc[:,0],bestidx_acc[:,1],marker='o',color='#33FF99')
    ax1.plot(bestidx_acc[:,0],bestidx_acc[:,1],color='#33FF99',alpha=1)

    ax1.scatter(worstidx_acc[:,0],worstidx_acc[:,1],marker='o',color='#ffddb7')
    ax1.plot(worstidx_acc[:,0],worstidx_acc[:,1],color='#ffddb7',alpha=1)
    
    ax1.tick_params(labelsize = 15)
    # ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.tick_params(pad = 5)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['left'].set_linewidth(0.5)
    ax1.set_xticks([-optimizer_change_point*scaling1,0]+[i for i in range(1, int(new_pop_acc[-1][0])) if i %50 == 0])#!!!
    ax1.set_xticklabels([-optimizer_change_point,0]+[i for i in range(1, int(new_pop_acc[-1][0])) if i % 50 == 0],fontsize = 8)#!!!
    ax.set_xticks([-optimizer_change_point*scaling1,0]+[i for i in range(1, int(new_pop_loss[-1][0])) if i % 50 == 0])#!!!
    ax.set_xticklabels([-optimizer_change_point,0]+[i for i in range(1, int(new_pop_loss[-1][0])) if i % 50 == 0],fontsize = 10)#!!!
    plt.yticks(fontsize = 20)

    # ax.set_ylim(1.232,1.245)
    # ax1.set_ylim(69.58,69.93)
    # ax1.vlines(0,ymin =69.58,ymax = 69.93,linestyles = 'dashed',colors = '#000000')
    # ax.vlines(0,ymin =1.232,ymax =1.25,linestyles = 'dashed',colors = '#000000')
    for tick in ax1.xaxis.get_ticklines():
        tick.set_markersize(3)
        tick.set_markeredgewidth(0.5)
    for tick in ax1.yaxis.get_ticklines():
        tick.set_markersize(3)
        tick.set_markeredgewidth(0.5)
#     plt.legend(loc='center')
    plt.savefig(savepath)
    if wandb is not None:
        wandb.log({'scatter': wandb.Image(fig)})

    fig_acc,ax_acc = plt.subplots(1,1,figsize = (5,5))
    ax_acc.plot([x*scaling1 if x<=0 else x for x in range(-optimizer_change_point,0)],adam_acc[:optimizer_change_point],color='#0557FA',alpha=1)
    ax_acc.scatter(x=new_pop_acc[:,0],y=new_pop_acc[:,1],marker='x',color='#F35858')
    ax_acc.scatter(x=old_pop_acc[:,0],y=old_pop_acc[:,1],marker='x',color='#007FFF')
    savepath2 = os.path.join(output_dir, 'acc.svg') 
    plt.savefig(savepath2)
    if wandb is not None:
        wandb.log({'acc': wandb.Image(fig_acc)})
    # wandb.finish()
    
    # plt.show()

def plot_paras(epoch, de_iter, max_de_iters, output_dir, plot_variables, wandb=None):
     u_freq_mean, u_f_mean, u_cr_mean, epoch_num, epoch_num_2, \
         cons_sim_list, l2_dist_list, lowest_dist_list, mean_dist_list, largest_dist_list, \
         L1_value_list, L2_value_list, succ_ls_list, p_stra_ls = plot_variables

     #________________"u_freq" 4 colors to decribe the 4 stratigy
     fig1, ax1 = plt.subplots(1,4,figsize = (15,4))
     colors = ['#c94733', '#3fab47', '#fddf8b', '#52b9d8']
     for i in range(4):
        ax1[0].plot(np.array(epoch_num)[:, i],np.array(u_freq_mean)[:, i],'x-',color=colors[i],label='p'+str(i+1))
        ax1[1].plot(np.array(epoch_num)[:, i],np.array(u_f_mean)[:, i],'x-',color=colors[i],label='p'+str(i+1))
        ax1[2].plot(np.array(epoch_num)[:, i],np.array(u_cr_mean)[:, i],'x-',color=colors[i],label='p'+str(i+1))
        ax1[3].plot(np.array(epoch_num)[:, i],np.array(p_stra_ls)[:, i],'x-',color=colors[i],label='p'+str(i+1))
     ax1[3].plot(np.array(epoch_num)[:, 0],np.array(p_stra_ls)[:, 4],'x-',label='trigo')
     # ax1.set_xlabel('epoch*max_de_iters + de_iter')
     ax1[0].set_xlabel('epoch')
     ax1[0].set_ylabel('u_freq')
     ax1[0].set_title('u_freq_mean')
     ax1[1].set_xlabel('epoch')
     ax1[1].set_ylabel('u_f')
     ax1[1].set_title('u_f_mean')
     ax1[2].set_xlabel('epoch')
     ax1[2].set_ylabel('u_cr')
     ax1[2].set_title('u_cr_mean')
     ax1[3].set_xlabel('epoch')
     for i in range(4):
        ax1[i].legend(fontsize=4)
     if wandb is not None:
        wandb.log({'mean_shade_setting': wandb.Image(fig1)})
     plt.savefig(os.path.join(output_dir,'mean_shade_setting.svg'))
     #____________"cons_sim" and "l2_dist"!!
     fig2, ax2 = plt.subplots(1, 3,figsize=(12,4))
     # ax4.plot([i for i in range(epoch*max_de_iters + de_iter)], cons_sim_list, label='cons_sim')
     # ax5.plot([i for i in range(epoch*max_de_iters + de_iter)], l2_dist_list,label='l2_dist')
     ax2[0].plot(np.array(epoch_num)[:, 0], cons_sim_list, label='cons_sim')
     ax2[1].plot(np.array(epoch_num)[:, 0], l2_dist_list,label='l2_dist')
     ax2[0].legend()
     ax2[1].legend()
     ax2[2].plot(np.array(epoch_num)[:, 0], lowest_dist_list, label='lowest_dist')
     ax2[2].plot(np.array(epoch_num)[:, 0], mean_dist_list, label='mean_dist')
     ax2[2].plot(np.array(epoch_num)[:, 0], largest_dist_list, label='largest_dist')
     ax2[2].legend()
     plt.savefig(os.path.join(output_dir,'cons_sim_and_l2_dist_eucl.svg'))
     if wandb is not None:
        wandb.log({'cons_sim_and_l2_dist_eucl': wandb.Image(fig2)})
     # _____________"L1_value L2_value"
     fig9, (ax9,ax10) = plt.subplots(2, 1,figsize=(12,6))
     # ax9.plot([i for i in range(epoch*max_de_iters + de_iter)], L1_value_list, label='L1_value')
     # ax10.plot([i for i in range(epoch*max_de_iters + de_iter)], L2_value_list, label='L2_value')
     pop_size = len(L2_value_list[0])
     for i in range(pop_size):
         ax9.scatter(np.array(epoch_num_2)[:, i],np.array(L1_value_list)[:, i],marker='x',color='#c94733')
         ax10.scatter(np.array(epoch_num_2)[:, i],np.array(L2_value_list)[:, i],marker='x',color='#c94733')
     # ax9.legend()
     # ax10.legend()
     plt.savefig(os.path.join(output_dir,'L1_value L2_value.svg'))
     if wandb is not None:
         wandb.log({'L1_value L2_value': wandb.Image(fig9)})

    # _____________"L1_value L2_value"
     fig3, ax3 = plt.subplots(2, 4,figsize=(15,8))
     # ax9.plot([i for i in range(epoch*max_de_iters + de_iter)], L1_value_list, label='L1_value')
     # ax10.plot([i for i in range(epoch*max_de_iters + de_iter)], L2_value_list, label='L2_value')
     pop_size = len(L2_value_list[0])
     succ_arr = np.array(succ_ls_list)
     for i in range(4):
         ax3[0,0].plot(np.array(epoch_num)[:, 0], succ_arr[:,2,9+i], label='strategy '+str(i+1))
         ax3[1,0].plot(np.array(epoch_num)[:, 0], succ_arr[:,3,9+i]/(succ_arr[:,2,9+i]+0.001), label='strategy '+str(i+1))
     ax3[0,0].plot(np.array(epoch_num)[:, 0], succ_arr[:,2,0], label='trigo')
     ax3[1,0].plot(np.array(epoch_num)[:, 0], succ_arr[:,3,0]/(succ_arr[:,2,0]+0.001), label='trigo')
     for i in range(4):
         ax3[0,1].plot(np.array(epoch_num)[:, 0], succ_arr[:,2,1+i], label='strategy '+str(i+1))
         ax3[1,1].plot(np.array(epoch_num)[:, 0], succ_arr[:,3,1+i]/(succ_arr[:,2,1+i]+0.001), label='strategy '+str(i+1))
     for i in range(4):
         ax3[0,2].plot(np.array(epoch_num)[:, 0], succ_arr[:,2,5+i], label='strategy '+str(i+1))
         ax3[1,2].plot(np.array(epoch_num)[:, 0], succ_arr[:,3,5+i]/(succ_arr[:,2,5+i]+0.001), label='strategy '+str(i+1))
     
     ax3[0,3].plot(np.array(epoch_num)[:, 0], np.sum(succ_arr[:,3,:],axis=1)/(np.sum(succ_arr[:,2,:],axis=1)+0.001), label='overall')
     ax3[0,3].legend(fontsize=4)
     ax3[1,3].plot(np.array(epoch_num)[:, 0], np.sum(succ_arr[:,3,:],axis=1), label='succ num')
     ax3[1,3].legend(fontsize=4)
     for i in range(3):
         ax3[0,i].legend(fontsize=4)
         ax3[1,i].legend(fontsize=4)
     plt.savefig(os.path.join(output_dir,'strategy_succ.svg'), bbox_inches="tight")
     if wandb is not None:
         wandb.log({'strategy_succ': wandb.Image(fig3)})



