#!/bin/bash
# python ../main_MDOT.py --max_epoch 500 --batch_size 1024 --logging True --section_ids "151507,151508" --num_class 7 --load_model False --num_hidden "512,30" --alpha_l 2 --lam 0.5 --loss_fn "sce" --mask_rate 0.05 --in_drop 0 --attn_drop 0 --remask_rate 0.05 
# python ../main_MDOT.py --max_epoch 1000 --logging True --section_ids "151507,151508" --num_class 7 --load_model False --num_hidden "512,32" --alpha_l 2 --lam 1 --loss_fn "sce" --mask_rate 0.05 --in_drop 0 --attn_drop 0 --remask_rate 0.05 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 1 2 3 4



## now loop through the above array

# 0920_1

## declare an array variable "151507,151508" "151508,151509" 
# declare -a arr=("151507,151508" "151508,151509" "151509,151510" "151673,151674" "151674,151675" "151675,151676")

# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1000 --logging False --section_ids "$i" --num_class 7 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.25 --in_drop 0 --attn_drop 0 --remask_rate 0.25 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0920_1" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 --num_remasking 2 --norm batchnorm
# done

# # "151669,151670" "151670,151671" "151671,151672"
# declare -a arr=("151669,151670" "151670,151671" "151671,151672")

# ## now loop through the above array
# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1000 --logging False --section_ids "$i" --num_class 5 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.25 --in_drop 0 --attn_drop 0 --remask_rate 0.25 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0920_1" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 --num_remasking 2 --norm batchnorm
# done

# # 0920_2
# declare -a arr=("151507,151508" "151508,151509" "151509,151510" "151673,151674" "151674,151675" "151675,151676")

# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1000 --logging False --section_ids "$i" --num_class 7 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.25 --in_drop 0 --attn_drop 0 --remask_rate 0.01 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0920_2" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 --num_remasking 2
# done

# # "151669,151670" "151670,151671" "151671,151672" 
# declare -a arr=("151669,151670" "151670,151671" "151671,151672")

# ## now loop through the above array
# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1000 --logging False --section_ids "$i" --num_class 5 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.25 --in_drop 0 --attn_drop 0 --remask_rate 0.01 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0920_2" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 --num_remasking 2
# done
#(hvg=7k)

# # 0920_3 hvg=8000
# declare -a arr=("151507,151508" "151508,151509" "151509,151510" "151673,151674" "151674,151675" "151675,151676")

# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1200 --logging False --section_ids "$i" --num_class 7 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.25 --in_drop 0 --attn_drop 0 --remask_rate 0.01 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0920_3" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 --num_remasking 2
# done

# # "151669,151670" "151670,151671" "151671,151672" 
# declare -a arr=("151669,151670" "151670,151671" "151671,151672")

# ## now loop through the above array
# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1200 --logging False --section_ids "$i" --num_class 5 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.25 --in_drop 0 --attn_drop 0 --remask_rate 0.01 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0920_3" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 --num_remasking 2
# done
# MaskedGEM 

# # 0920_4 hvg=7000 increase mask rate
# declare -a arr=("151507,151508" "151508,151509" "151509,151510" "151673,151674" "151674,151675" "151675,151676")

# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1000 --logging False --section_ids "$i" --num_class 7 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.35 --in_drop 0 --attn_drop 0 --remask_rate 0.01 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0920_4" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 --num_remasking 2 --hvgs 7000
# done

# # "151669,151670" "151670,151671" "151671,151672" 
# declare -a arr=("151669,151670" "151670,151671" "151671,151672")

# ## now loop through the above array
# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1000 --logging False --section_ids "$i" --num_class 5 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.35 --in_drop 0 --attn_drop 0 --remask_rate 0.01 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0920_4" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 --num_remasking 2 --hvgs 7000
# done

# # 0920_5 hvg=7000 increase mask rate, also increase backbone capacity
# declare -a arr=("151507,151508" "151508,151509" "151509,151510" "151673,151674" "151674,151675" "151675,151676")

# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1000 --logging False --section_ids "$i" --num_class 7 --load_model False --num_hidden "800,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.35 --in_drop 0 --attn_drop 0 --remask_rate 0.01 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0920_5" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 --num_remasking 2 --hvgs 7000
# done

# # "151669,151670" "151670,151671" "151671,151672" 
# declare -a arr=("151669,151670" "151670,151671" "151671,151672")

# ## now loop through the above array
# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1000 --logging False --section_ids "$i" --num_class 5 --load_model False --num_hidden "800,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.35 --in_drop 0 --attn_drop 0 --remask_rate 0.01 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0920_5" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 --num_remasking 2 --hvgs 7000
# done

# # 0920_6 reproduce setting 2(0920 2)
# declare -a arr=("151507,151508" "151508,151509" "151509,151510" "151673,151674" "151674,151675" "151675,151676")

# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1000 --logging False --section_ids "$i" --num_class 7 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.25 --in_drop 0 --attn_drop 0 --remask_rate 0.01 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0920_6" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 1 2 3 4 --num_remasking 2 --hvgs 7000
# done

# # "151669,151670" "151670,151671" "151671,151672" 
# declare -a arr=("151669,151670" "151670,151671" "151671,151672")

# ## now loop through the above array
# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1000 --logging False --section_ids "$i" --num_class 5 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.25 --in_drop 0 --attn_drop 0 --remask_rate 0.01 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0920_6" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 1 2 3 4 --num_remasking 2 --hvgs 7000
# done

# # 0921_1
# declare -a arr=("151507,151508" "151508,151509" "151509,151510" "151673,151674" "151674,151675" "151675,151676")

# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1000 --logging False --section_ids "$i" --num_class 7 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.25 --in_drop 0 --attn_drop 0 --remask_rate 0.25 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0921_1" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 1 --num_remasking 3 --hvgs 7000
# done

# # "151669,151670" "151670,151671" "151671,151672" 
# declare -a arr=("151669,151670" "151670,151671" "151671,151672")

# ## now loop through the above array
# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1000 --logging False --section_ids "$i" --num_class 5 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.25 --in_drop 0 --attn_drop 0 --remask_rate 0.25 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0921_1" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 1 --num_remasking 3 --hvgs 7000
# done

# 0921_2 decrease second stage epochs
declare -a arr=("151507,151508" "151508,151509" "151509,151510" "151673,151674" "151674,151675" "151675,151676")

for i in "${arr[@]}"
do
    echo "$i"
    python ../main_MDOT.py --max_epoch 1500 --logging False --section_ids "$i" --num_class 7 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.15 --in_drop 0 --attn_drop 0 --remask_rate 0.15 \
                       --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0922_1" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
                       --seeds 0 1 3 5 7 --num_remasking 2 --hvgs 7000
done

# "151669,151670" "151670,151671" "151671,151672" 
declare -a arr=("151669,151670" "151670,151671" "151671,151672")

## now loop through the above array
for i in "${arr[@]}"
do
    echo "$i"
    python ../main_MDOT.py --max_epoch 1000 --logging False --section_ids "$i" --num_class 5 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.15 --in_drop 0 --attn_drop 0 --remask_rate 0.15 \
                       --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0922_1" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
                       --seeds 0 1 3 5 7 --num_remasking 2 --hvgs 7000
done

# 0921_3 back to mse
# declare -a arr=("151507,151508" "151508,151509" "151509,151510" "151673,151674" "151674,151675" "151675,151676")

# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1000 --logging False --section_ids "$i" --num_class 7 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "mse" --mask_rate 0.01 --in_drop 0 --attn_drop 0 --remask_rate 0.01 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0921_3" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 1 2 --num_remasking 1 --hvgs 5000 --dataset DLPFC
# done

# # "151669,151670" "151670,151671" "151671,151672" 
# declare -a arr=("151669,151670" "151670,151671" "151671,151672")

# ## now loop through the above array
# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 1000 --logging False --section_ids "$i" --num_class 5 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "mse" --mask_rate 0.01 --in_drop 0 --attn_drop 0 --remask_rate 0.01 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_0921_3" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 1 2 --num_remasking 1 --hvgs 5000 --dataset DLPFC
# done