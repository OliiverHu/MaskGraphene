<h1> MaskGraphene: Advancing clustering, joint embedding, and batch correction for spatial transcriptomics using graph-based self-supervised learning </h1>

Implementation for Recomb 2024 paper:  [MaskGraphene]().
<img src="/figs/ppl.png">


<h2>Dependencies </h2>

* Python >= 3.9
* [Pytorch](https://pytorch.org/) == 2.0.1
* anndata==0.9.2
* h5py==3.9.0
* hnswlib==0.7.0
* igraph==0.10.8
* matplotlib==3.6.3
* paste-bio==1.4.0
* POT==0.9.1
* rpy2==3.5.14
* scanpy==1.9.1
* umap-learn==0.5.4
* wandb
* pyyaml == 5.4.1

<h2>Installation</h2>

```bash
conda create -n MaskGraphene python=3.9 

pip install -r requirements.txt
```

For DGL package, please refer to [link](https://www.dgl.ai/pages/start.html)

```bash
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

<h2>Quick Start </h2>

For quick start, you could run the scripts: 

**DLPFC 151507/151508**

```bash
python ./maskgraphene_main.py --max_epoch 1500 --max_epoch_triplet 700 --logging False --section_ids "151507,151508" --num_class 7 --load_model False --num_hidden "512,32" --alpha_l 2 --lam 1 --loss_fn "sce" 
                              --mask_rate 0.25 --in_drop 0 --attn_drop 0 --remask_rate 0.25
                              --exp_fig_dir "./" --h5ad_save_dir "./" --st_data_dir "./benchmarking_data/DLPFC12"
                              --seeds 2023 --num_remasking 1 --hvgs 6000 --dataset DLPFC --consecutive_prior 1
```

Supported ST datasets:

<!-- * mini batch node classification:  `ogbn-arxiv`, `ogbn-products`, `mag-scholar-f`, `ogbn-papers100M` -->
* 10x Visium: [`DLPFC`](), [`Mouse Sagittal Brain`]()
* Others: [`mouse Hypothalamus`](), [`Embryo`]()



<h1> Citing </h1>

Currently under review

<!-- ```

``` -->
