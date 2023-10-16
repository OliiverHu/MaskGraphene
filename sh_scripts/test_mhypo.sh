# 0921_1
# declare -a arr=(" -0.04,-0.09" " -0.09,-0.14" " -0.14,-0.19" " -0.19,-0.24")
# declare -a arr=(" -0.04,-0.09" " -0.09,-0.14" " -0.14,-0.19" " -0.19,-0.24")

# for i in "${arr[@]}"
# do
#     echo "$i"
#     python ../main_MDOT.py --max_epoch 2500 --max_epoch_triplet 500 --logging False --section_ids "$i" --num_class 8 --load_model False --num_hidden "512,32" --alpha_l 3 --lam 1 --loss_fn "sce" --mask_rate 0.60 --in_drop 0 --attn_drop 0 --remask_rate 0.40 \
#                        --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_mhypo1005_3" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/mHypothalamus" \
#                        --seeds 2023 --num_remasking 2 --hvgs 0 --dataset mHypothalamus
# done
# # in this setting, also set knn thres to 40 to limit nn numbers.

# 1005 4
declare -a arr=(" -0.04,-0.09" " -0.09,-0.14" " -0.14,-0.19" " -0.19,-0.24")

for i in "${arr[@]}"
do
    echo "$i"
    python ../main_MDOT.py --max_epoch 3000 --max_epoch_triplet 1000 --logging False --section_ids "$i" --num_class 8 --load_model False --num_hidden "512,32" --alpha_l 3 --lam 1 --loss_fn "sce" --mask_rate 0.70 --in_drop 0 --attn_drop 0 --remask_rate 0.70 \
                       --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_mhypo1005_4" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/mHypothalamus" \
                       --seeds 2023 --num_remasking 3 --hvgs 0 --dataset mHypothalamus
done
# in this setting, also set knn thres to 40 to limit nn numbers.