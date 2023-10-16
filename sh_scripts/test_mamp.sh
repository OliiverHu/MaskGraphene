# 0925_1
declare -a arr=("MA,MP")

for i in "${arr[@]}"
do
    echo "$i"
    python ../main_GM_nogt.py --max_epoch 1000 --max_epoch_triplet 500 --logging False --section_ids "$i" --num_class 30 --load_model False --num_hidden "512,32" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.25 --in_drop 0 --attn_drop 0 --remask_rate 0.25 \
                       --exp_fig_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/ST_exps_mamp0927_1" --h5ad_save_dir "/home/yunfei/spatial_dl_integration/GraphMAE2/st_dataset" --st_data_dir "/home/yunfei/spatial_benchmarking/benchmarking_data/mMAMP" \
                       --seeds 0 1 3 5 666 7 9 11 13 2023 --num_remasking 1 --hvgs 8000 --dataset mMAMP
done