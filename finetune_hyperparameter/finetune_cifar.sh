#finetune 
    CUDA_VISIBLE_DEVICES=2 \
    python -m torch.distributed.launch --nproc_per_node=1 --master_port='18802' \
        finetune.py --batch_size 128 \
            --val_batch_size 128 \
            --slice_len 0 \
            --warmup 0 \
            --opt 'sgd' \
            --means 1.0 \
            --lamb 0.05 \
            --sched 'cosine' \
            --epochs 200 \
            --model 'sew_34' \
            --resume "/home/guodong/runhua/cifar_finetune/spe_snn_finetune_cifar/output/train/20231022-143814-sew34_cifar10_sgd_lr_1e-1_sm_0_wd5e-4_300epoch_T4/sew34_cifar10_sgd_lr_1e-1_sm_0_wd5e-4_300epoch_T4_299.pt" \
            --connect_f 'ADD' \
            --zero_init_residual \
            --lr 1e-1 \
            --lr_min 0 \
            --smoothing 0 \
            --mixup 0.1 \
            --aa 'rand-m11-n3' \
            --re_prob 0 \
            --amp \
            --T 4 \
            --data '/home/guodong/runhua/data' \
            --dataset_name 'cifar100' \
            --num_classes 10 \
            --output /home/guodong/runhua/cifar_finetune/spe_snn_finetune_cifar/output \
            --exp_name "sew34(from_cifar10_train)_cifar100_sgd_lr_1e-1_wd5e-4_300epoch_smoothing_0_mixup_0.1_T4_1_1_1_1_epoch_200_aa_rand-m11-n3" \
            --log_dir "log_out/log/sew34(from_cifar10_train)_cifar100_sgd_lr_1e-1_wd5e-4_300epoch_smoothing_0_mixup_0.1_1_1_1_1_epoch_200_aa_rand-m11-n3.txt" \
            --weight_decay 5e-4 \
            --time_ratio 1 1 1 1\
            --cal_ratio_loss True \
            >& log_out/sew34_from_cifar10_train_cifar100_sgd_lr_1e-1_wd5e-4_300epoch_smoothing_0_mixup_0.1_1_1_1_1_epoch_200_aa_rand-m11-n3.txt &
            # --only_test \
            # --TET True \

#train no resume
#     CUDA_VISIBLE_DEVICES=0 \
#     python -m torch.distributed.launch --nproc_per_node=1 --master_port='18805' \
#         finetune.py --batch_size 128 \
#             --val_batch_size 128 \
#             --slice_len 0 \
#             --warmup 0 \
#             --opt 'sgd' \
#             --means 1.0 \
#             --lamb 0.05 \
#             --sched 'cosine' \
#             --epochs 200 \
#             --model 'sew_18' \
#             --connect_f 'ADD' \
#             --zero_init_residual \
#             --lr 1e-1 \
#             --lr_min 0 \
#             --smoothing 0 \
#             --mixup 0 \
#             --aa None \
#             --amp \
#             --T 6 \
#             --data '/home/guodong/runhua/data' \
#             --dataset_name 'cifar100' \
#             --num_classes 100 \
#             --output /data/guodong/runhua/0908/spe_snn_finetune_cifar/output \
#             --exp_name "sew_18_T_6" \
#             --log_dir "log_out/log/sew_18_T_6.txt" \
#             --weight_decay 5e-4 \
#             >& log_out/sew_18_T_6_all.txt &
    #         # --TET True \
            # --resume "/data/guodong/runhua/0908/spe_snn_finetune_cifar/output/train/20230909-003343-resnet19_cifar100_sgd_1e-1_weight_decay_5e-4_200epoch_T_6_TET/resnet19_cifar100_sgd_1e-1_weight_decay_5e-4_200epoch_T_6_TET_199.pt" \
            # --time_ratio 1 1 1 1 1 1\
            # --cal_ratio_loss True \
    #         # --only_test \
#-----only test
    #     CUDA_VISIBLE_DEVICES=0 \
    # python -m torch.distributed.launch --nproc_per_node=1 --master_port='18811' \
    #     finetune.py --batch_size 128 \
    #         --val_batch_size 128 \
    #         --slice_len 0 \
    #         --warmup 0 \
    #         --opt 'sgd' \
    #         --means 1.0 \
    #         --lamb 0.05 \
    #         --sched 'cosine' \
    #         --epochs 50 \
    #         --model 'sew_18' \
    #         --resume "/home/guodong/cl/SEW18_T4_C100best_model.pt" \
    #         --connect_f 'ADD' \
    #         --zero_init_residual \
    #         --lr 1e-3 \
    #         --lr_min 0 \
    #         --smoothing 0 \
    #         --mixup 0 \
    #         --aa None \
    #         --amp \
    #         --T 6 \
    #         --data '/home/guodong/runhua/data' \
    #         --dataset_name 'cifar100' \
    #         --num_classes 100 \
    #         --output /data/guodong/runhua/0908/spe_snn_finetune_cifar/output \
    #         --exp_name "only_test" \
    #         --log_dir "log_out/log/only_test.txt" \
    #         --weight_decay 5e-4 \
    #         --time_ratio 1 1 1 1 1 1\
    #         --cal_ratio_loss True \
    #         --only_test \
    #         # >& log_out/only_test.txt &
    #         # --TET True \


# for i in {5e-5,5e-3,1e-5,1e-4}
# do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     python -m torch.distributed.launch --nproc_per_node=4 --master_port='18808' \
#         finetune_1.py --batch_size 32 \
#             --val_batch_size 64 \
#             --slice_len 0 \
#             --warmup 0 \
#             --opt 'sgd' \
#             --sched 'cosine' \
#             --epochs 30 \
#             --model 'resnet19' \
#             --lr 1e-5 \
#             --smoothing 0 \
#             --mixup 0 \
#             --aa None \
#             --amp \
#             --T 10 \
#             --data '/home/guodong/runhua/data' \
#             --dataset_name 'cifar10' \
#             --num_classes 10 \
#             --output /home/guodong/runhua/spe_snn_finetune_cifar/output \
#             --exp_name "resnet19_cifar10_lr_1e-5_resume_T_10_weight_decay_$i" \
#             --resume '/home/guodong/runhua/model/T10_Dcifar10_Aresnet19_ce_ckpt_0300.pth.tar' \
#             --log_dir "log_out/log/resnet19_cifar10_lr_1e-5_resume_T_10_weight_decay_$i.txt" \
#             --weight_decay $i \
#             >& log_out/resnet19_cifar10_lr_1e-5_resume_T_10_weight_decay_$i.txt 

# done
        # --only_test \

        # --aa 
# Experiment names:
# 1. debug
# 2. only_test
# 3. epoch.3_lr.1e-3_adamW
# 4. epoch.6_lr.1e-3_sgd
# 5. epoch.10_lr.1e-3_sgd_bs512_mix0.2
# 6. epoch.20_lr.0.01_sgd_bs512_sm0.1
# "rand-m14-n1"

# sm 0
# mix 0
# lr 1e-4
# wd 2e-5
# epochs 20
# aa 1，2，3，4