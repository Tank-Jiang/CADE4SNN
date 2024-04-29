# -----Test


CUDA_VISIBLE_DEVICES=3 \
python -m torch.distributed.launch --nproc_per_node=1 --master_port='24683' \
    main_cosde.py --batch_size 128 \
        --val_batch_size 192 \
        --slice_len 0 \
        --de_slice_len 0 \
        --de_epochs 100 \
        --de_batch_size 512 \
        --popsize 8 \
        --model 'sew_34' \
        --amp \
        --num-gpu 1 \
        --data '/home/runhua/data' \
        --num_classes 100 \
        --trig_perc 0.01 \
        --f_init 1e-9 \
        --cr_init 1e-9 \
        --cr_clip 0.001 \
        --f_clip 0.001 \
        --pop_init '/home/runhua/1215/spe/utils/data/1215_above_77.txt' \
        --exp_name cr_cos_f_cos_1e-9_1215_above_77.txt \
        --output /home/runhua/1215/spe/output \
        --log_dir '/home/runhua/1215/spe/output/cr_cos_f_cos_1e-9_1215_above_77.txt' \
        &> /home/runhua/1215/spe/log_out/cr_cos_f_cos_1e-9_1215_above_77.txt &
        # --test_ood \
        # --ood_path '/home/guodong/runhua/ood_dataset' \
# -----Test

# CUDA_VISIBLE_DEVICES=0 \
# python -m torch.distributed.launch --nproc_per_node=1 --master_port='24680' \
#     main_cosde.py --batch_size 128 \
#         --val_batch_size 192 \
#         --slice_len 0 \
#         --de_slice_len 0 \
#         --de_epochs 100 \
#         --de_batch_size 512 \
#         --popsize 8 \
#         --model 'sew_34' \
#         --amp \
#         --num-gpu 1 \
#         --data '/home/runhua/0814/data' \
#         --num_classes 100 \
#         --trig_perc 0.01 \
#         --f_init 1e-5 \
#         --cr_init 1e-5 \
#         --cr_clip 0.001 \
#         --f_clip 0.001 \
#         --pop_init '/home/runhua/1129/spe_final_version_snn_good_result/utils/data/snn_from_cifar10_pretrain_change_hyperparamter_finetune.txt' \
#         --exp_name hyperparamter_test_de_stratgy_3_f_init_1e-5_cr_init_1e-5_T_40 \
#         --output /home/runhua/1129/spe_final_version_snn_good_result/output \
#         --log_dir '/home/runhua/1129/spe_final_version_snn_good_result/output/hyperparamter_test_de_stratgy_3_f_init_1e-5_cr_init_1e-5_T_40.txt' \
#         &> /home/runhua/1129/spe_final_version_snn_good_result/log_out/hyperparamter_test_de_stratgy_3_f_init_1e-5_cr_init_1e-5_T_40.txt &
#         # --test_ood \
#         # --ood_path '/home/guodong/runhua/ood_dataset' \
       
# experiment 1: --pop_init '/home/runhua/1129/spe_final_version_snn_good_result/utils/data/snn_cifar100_divide_5.txt' \
# experiment 2: --pop_init '/home/runhua/1129/spe_final_version_snn_good_result/utils/data/snn_change_timeratio_10_or_50.txt' \ 
# experiment 3: --pop_init '/home/runhua/1129/spe_final_version_snn_good_result/utils/data/snn_train_nopretrain_time_ratio_change_fineturn.txt' \