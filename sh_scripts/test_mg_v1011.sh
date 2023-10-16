# declare -a arr=("151507,151508" "151508,151509" "151509,151510" "151673,151674" "151674,151675" "151675,151676")

# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../maskgraphene_main.py --max_epoch 2000 --max_epoch_triplet 700 --logging False --section_ids "$i" --num_class 7 --load_model False --num_hidden "512,32" --alpha_l 2 --lam 1 --loss_fn "sce" --mask_rate 0.3 --in_drop 0 --attn_drop 0 --remask_rate 0.3 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/MG_exps_1012_1" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 5 42 114514 2023 --num_remasking 1 --hvgs 7500 --dataset DLPFC --consecutive_prior 1
# done

# # "151669,151670" "151670,151671" "151671,151672" 
# declare -a arr=("151669,151670" "151670,151671" "151671,151672")

# ## now loop through the above array
# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../maskgraphene_main.py --max_epoch 2000 --max_epoch_triplet 700 --logging False --section_ids "$i" --num_class 5 --load_model False --num_hidden "512,32" --alpha_l 2 --lam 1 --loss_fn "sce" --mask_rate 0.3 --in_drop 0 --attn_drop 0 --remask_rate 0.3 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/MG_exps_1012_1" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
#                        --seeds 0 5 42 114514 2023 --num_remasking 1 --hvgs 7500 --dataset DLPFC --consecutive_prior 1
# done


# ####### mouse hypothalamus
# declare -a arr=(" -0.04,-0.09" " -0.09,-0.14" " -0.14,-0.19" " -0.19,-0.24")

# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../maskgraphene_main.py --max_epoch 3000 --max_epoch_triplet 1000 --logging False --section_ids "$i" --num_class 8 --load_model False --num_hidden "512,32" --alpha_l 3 --lam 1 --loss_fn "sce" --mask_rate 0.70 --in_drop 0 --attn_drop 0 --remask_rate 0.70 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/MG_exps_1012_1" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/mHypothalamus" \
#                        --seeds 0 5 42 114514 2023 --num_remasking 3 --hvgs 0 --dataset mHypothalamus --consecutive_prior 1
# done

declare -a arr=("151507,151508" "151508,151509" "151509,151510" "151673,151674" "151674,151675" "151675,151676")

for i in "${arr[@]}"
do
    echo "$i"
    python ../maskgraphene_main.py --max_epoch 1500 --max_epoch_triplet 700 --logging False --section_ids "$i" --num_class 7 --load_model False --num_hidden "512,32" --alpha_l 2 --lam 1 --loss_fn "sce" --mask_rate 0.25 --in_drop 0 --attn_drop 0 --remask_rate 0.25 \
                       --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/MG_exps_1014_1" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
                       --seeds 0 5 42 114514 2023 --num_remasking 1 --hvgs 6500 --dataset DLPFC --consecutive_prior 1
done

# "151669,151670" "151670,151671" "151671,151672" 
declare -a arr=("151669,151670" "151670,151671" "151671,151672")

## now loop through the above array
for i in "${arr[@]}"
do
    echo "$i"
    python ../maskgraphene_main.py --max_epoch 1500 --max_epoch_triplet 700 --logging False --section_ids "$i" --num_class 5 --load_model False --num_hidden "512,32" --alpha_l 2 --lam 1 --loss_fn "sce" --mask_rate 0.25 --in_drop 0 --attn_drop 0 --remask_rate 0.25 \
                       --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/MG_exps_1014_1" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12" \
                       --seeds 0 5 42 114514 2023 --num_remasking 1 --hvgs 6500 --dataset DLPFC --consecutive_prior 1
done


####### mouse hypothalamus
declare -a arr=(" -0.04,-0.09" " -0.09,-0.14" " -0.14,-0.19" " -0.19,-0.24")

for i in "${arr[@]}"
do
    echo "$i"
    python ../maskgraphene_main.py --max_epoch 3000 --max_epoch_triplet 1000 --logging False --section_ids "$i" --num_class 8 --load_model False --num_hidden "512,32" --alpha_l 3 --lam 1 --loss_fn "sce" --mask_rate 0.70 --in_drop 0 --attn_drop 0 --remask_rate 0.70 \
                       --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/MG_exps_1014_1" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/mHypothalamus" \
                       --seeds 0 5 42 114514 2023 --num_remasking 3 --hvgs 0 --dataset mHypothalamus --consecutive_prior 1
done