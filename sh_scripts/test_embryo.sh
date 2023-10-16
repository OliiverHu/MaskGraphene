# 0925_1
# declare -a arr=(" -0.04,-0.09" " -0.09,-0.14" " -0.14,-0.19" " -0.19,-0.24")
declare -a arr=("11.5_E1S1,12.5_E1S1")
# 9.5/11.5 = 23
# 10.5/11.5 = 19
# 11.5/12.5 = 30

for i in "${arr[@]}"
do
    echo "$i"
    python ../main_GM_nogt.py --max_epoch 1000 --max_epoch_triplet 500 --logging False --section_ids "$i" --num_class 15 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.15 --in_drop 0 --attn_drop 0 --remask_rate 0.15 \
                       --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_embryo0928_1" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/Embryo" \
                       --seeds 0 1 3 5 666 7 9 11 13 2023 --num_remasking 1 --hvgs 5000 --dataset Embryo --device 3
done
