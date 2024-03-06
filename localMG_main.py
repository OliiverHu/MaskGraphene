import logging
import numpy as np
from tqdm import tqdm
import torch
import anndata
import dgl
from datasets.st_loading_utils import load_DLPFC, create_dictionary_mnn, load_mHypothalamus
from datasets.data_proc import Cal_Spatial_Net
import scipy
import paste
import ot
import os
import pickle

from utils import (
    build_args_ST,
    create_optimizer,
    set_random_seed,
    WandbLogger,
    TBLogger,
    get_current_lr,
)
from datasets.data_proc import load_ST_dataset_hard
from datasets.st_loading_utils import create_dictionary_otn, mclust_R, gmm_scikit, cal_layer_based_alignment_result, visualization_umap_spatial
from models import build_model_ST
import scanpy as sc
from sklearn.metrics import adjusted_rand_score as ari_score


def cal_layer_based_alignment_result(alignment, s1, s2):
    labels = []
    labels.extend(s1.obs['original_clusters'])
    labels.extend(s2.obs['original_clusters'])

    res = []
    l_dict = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
    cnt0 = 0
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    cnt6 = 0
    for i, elem in enumerate(alignment):
        if labels[i] == '-1' or labels[elem.argmax() + alignment.shape[0]] == '-1':
            continue
        if l_dict[labels[i]] == l_dict[labels[elem.argmax() + alignment.shape[0]]]:
            cnt0 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 1:
            cnt1 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 2:
            cnt2 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 3:
            cnt3 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 4:
            cnt4 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 5:
            cnt5 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 6:
            cnt6 += 1
    #print(alignment.shape[0])
    #print(cnt0/alignment.shape[0], cnt1/alignment.shape[0], cnt2/alignment.shape[0], cnt3/alignment.shape[0], cnt4/alignment.shape[0], cnt5/alignment.shape[0], cnt6/alignment.shape[0])
    #res.extend([cnt0/alignment.shape[0], cnt1/alignment.shape[0], cnt2/alignment.shape[0], cnt3/alignment.shape[0], cnt4/alignment.shape[0], cnt5/alignment.shape[0], cnt6/alignment.shape[0]])
    res.extend([cnt0, cnt1, cnt2, cnt3, cnt4, cnt5, cnt6])
    return res


def localOT_loader(section_ids=["151507", "151508"], dataname="DLPFC", hvgs=5000, st_data_dir="./", hard_links=None):
    # hard links is a mapping matrix (2d numpy array with the size of #slice1 spot by #slice2 spot)
    if dataname == "DLPFC":
        Batch_list = []
        adj_list = []
        for section_id in section_ids:
            ad_ = load_DLPFC(root_dir=st_data_dir, section_id=section_id)
            ad_.var_names_make_unique(join="++")
        
            # make spot name unique
            ad_.obs_names = [x+'_'+section_id for x in ad_.obs_names]
            
            # Constructing the spatial network
            Cal_Spatial_Net(ad_, rad_cutoff=150) # the spatial network are saved in adata.uns[‘adj’]
            
            # Normalization
            sc.pp.highly_variable_genes(ad_, flavor="seurat_v3", n_top_genes=hvgs)
            sc.pp.normalize_total(ad_, target_sum=1e4)
            sc.pp.log1p(ad_)
            ad_ = ad_[:, ad_.var['highly_variable']]

            adj_list.append(ad_.uns['adj'])
            Batch_list.append(ad_)
        adata_concat = anndata.concat(Batch_list, label="slice_name", keys=section_ids, uns_merge="same")
        adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
        adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

        adj_concat = np.asarray(adj_list[0].todense())
        for batch_id in range(1,len(section_ids)):
            adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
        
        """if hard links is not empty"""
        if hard_links != None:
            for i in range(hard_links.shape[0]):
                for j in range(hard_links.shape[1]):
                    if hard_links[i][j] > 0:
                        adj_concat[i][j+hard_links.shape[0]] = 1
                        adj_concat[j+hard_links.shape[0]][i] = 1

        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph.ndata["feat"] = torch.tensor(adata_concat.X.todense())
        num_features = graph.ndata["feat"].shape[1]
    elif dataname == "mHypothalamus":
        Batch_list = []
        adj_list = []
        for section_id in section_ids:
            ad_ = load_mHypothalamus(root_dir=st_data_dir, section_id=section_id)
            ad_.var_names_make_unique(join="++")
        
            # make spot name unique
            ad_.obs_names = [x+'_'+section_id for x in ad_.obs_names]
            
            # Constructing the spatial network
            Cal_Spatial_Net(ad_, rad_cutoff=35) # the spatial network are saved in adata.uns[‘adj’]
            
            # Normalization
            sc.pp.normalize_total(ad_, target_sum=1e4)
            sc.pp.log1p(ad_)

            adj_list.append(ad_.uns['adj'])
            Batch_list.append(ad_)
        adata_concat = anndata.concat(Batch_list, label="slice_name", keys=section_ids, uns_merge="same")
        adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
        adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

        adj_concat = np.asarray(adj_list[0].todense())
        for batch_id in range(1,len(section_ids)):
            adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
        
        """if hard links is not empty"""
        if hard_links != None:
            for i in range(hard_links.shape[0]):
                for j in range(hard_links.shape[1]):
                    if hard_links[i][j] > 0:
                        adj_concat[i][j+hard_links.shape[0]] = 1
                        adj_concat[j+hard_links.shape[0]][i] = 1

        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph.ndata["feat"] = torch.tensor(adata_concat.X).float()
        num_features = graph.ndata["feat"].shape[1]
    else:
        raise NotImplementedError
    return graph, num_features, adata_concat


def run_MG_aligner(graph, model, device, ad_concat, section_ids, max_epoch, max_epoch_triplet, optimizer, scheduler, logger, use_mnn=False):
    x = graph.ndata["feat"]
    model.to(device)
    graph = graph.to(device)
    x = x.to(device)

    """training"""
    target_nodes = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
    epoch_iter = tqdm(range(max_epoch))

    print("training local clusters ... ")
    for epoch in epoch_iter:
        model.train()
        # print(type(x), type(graph))
        loss = model(graph, x, targets=target_nodes)

        loss_dict = {"loss": loss.item()}
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.log(loss_dict, step=epoch)
    
    with torch.no_grad():
        embedding = model.embed(graph, x)
    ad_concat.obsm["maskgraphene"] = embedding.cpu().detach().numpy()
    
    if use_mnn:
        mnn_dict = create_dictionary_mnn(ad_concat, use_rep="maskgraphene", batch_name='batch_name', k=50, verbose = 1, iter_comb=None)

        anchor_ind = []
        positive_ind = []
        negative_ind = []
        for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
            batchname_list = ad_concat.obs['batch_name'][mnn_dict[batch_pair].keys()]
            #             print("before add KNN pairs, len(mnn_dict[batch_pair]):",
            #                   sum(adata_new.obs['batch_name'].isin(batchname_list.unique())), len(mnn_dict[batch_pair]))

            cellname_by_batch_dict = dict()
            for batch_id in range(len(section_ids)):
                cellname_by_batch_dict[section_ids[batch_id]] = ad_concat.obs_names[
                    ad_concat.obs['batch_name'] == section_ids[batch_id]].values

            anchor_list = []
            positive_list = []
            negative_list = []
            for anchor in mnn_dict[batch_pair].keys():
                anchor_list.append(anchor)
                ## np.random.choice(mnn_dict[batch_pair][anchor])
                positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                positive_list.append(positive_spot)
                section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                negative_list.append(
                    cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])

            batch_as_dict = dict(zip(list(ad_concat.obs_names), range(0, ad_concat.shape[0])))
            anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
            positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
            negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

        epoch_iter = tqdm(range(max_epoch_triplet))
        for epoch in epoch_iter:
            model.train()
            optimizer.zero_grad()

            _loss = model(graph, x, targets=target_nodes)
            if epoch % 100 == 0 or epoch == 500:
                with torch.no_grad():
                    z = model.embed(graph, x)
                
                anchor_arr = z[anchor_ind,]
                positive_arr = z[positive_ind,]
                negative_arr = z[negative_ind,]

            triplet_loss = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')
            tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

            loss = _loss + tri_output
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            loss_dict = {"loss": loss.item()}
            epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
            if logger is not None:
                loss_dict["lr"] = get_current_lr(optimizer)
                logger.log(loss_dict, step=epoch)
        
            # z = model.embed(graph, x)
        with torch.no_grad():
            embedding = model.embed(graph, x)
        ad_concat.obsm["maskgraphene_mnn"] = embedding.cpu().detach().numpy()

    """calculate ARI & umap & viz"""
    if use_mnn:
        mclust_R(ad_concat, modelNames='EEE', num_cluster=args.num_class, used_obsm='maskgraphene_mnn')
    else:
        mclust_R(ad_concat, modelNames='EEE', num_cluster=args.num_class, used_obsm='maskgraphene')

    ad_temp = ad_concat[ad_concat.obs['original_clusters']!='unknown']
    
    
    Batch_list = []
    for section_id in section_ids:
        ad__ = ad_temp[ad_temp.obs['batch_name'] == section_id]
        Batch_list.append(ad__)
        print(section_id)
        print('mclust, ARI = %01.3f' % ari_score(ad__.obs['original_clusters'], ad__.obs['mclust']))
        # print("using mclust")
    
    return Batch_list


def localMG(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_triplet = args.max_epoch_triplet

    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate
    # encoder = args.encoder
    # decoder = args.decoder
    is_consecutive = args.consecutive_prior

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    # linear_prob = args.linear_prob
    # load_model = args.load_model
    logs = False
    use_scheduler = args.scheduler

    """mid files save path"""
    exp_fig_dir = args.exp_fig_dir
    # h5ad_save_dir = args.h5ad_save_dir
    st_data_dir = args.st_data_dir

    """ST loading"""
    section_ids = args.section_ids.lstrip().split(",")
    

    model_dir = "checkpoints"
    os.makedirs(model_dir, exist_ok=True)
    # print(logs)
    

    # acc_list = []
    # estp_acc_list = []
    ari_1 = []
    ari_2 = []
    if not os.path.exists(os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids))):
        os.makedirs(os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids)))
    exp_fig_dir = os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids))

    # use_cached_pi = True
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = WandbLogger(log_path=f"{dataset_name}_{'_'.join(section_ids)}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}__wd_{weight_decay}__{encoder_type}_{decoder_type}", project="M-DOT", args=args)
            # logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}__wd_{weight_decay}__{encoder_type}_{decoder_type}")
        else:
            logger = None
        # model_name = f"{encoder}_{decoder}_{num_hidden}_{num_layers}_{dataset_name}_{'_'.join(section_ids)}_{args.mask_rate}_{num_hidden[::-1]}_checkpoint.pt"
        
        
        """
        """
        """
        STAGE 0
        """
        """calculate hard links if possible, otherwise, skip this stage"""
        if is_consecutive == 1:
            graph, num_features, ad_concat = localOT_loader(section_ids=section_ids, hvgs=args.hvgs, st_data_dir=st_data_dir, dataname=dataset_name)
            args.num_features = num_features
            print(args)
            model_local_ot = build_model_ST(args)
            print(model_local_ot)
            model_local_ot.to(device)
            optimizer = create_optimizer(optim_type, model_local_ot, lr, weight_decay)

            if use_scheduler:
                logging.info("Use scheduler")
                scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
            else:
                scheduler = None
            
            batchlist_ = run_MG_aligner(graph, model_local_ot, device, ad_concat, section_ids, max_epoch=max_epoch, max_epoch_triplet=max_epoch_triplet, optimizer=optimizer, scheduler=scheduler, logger=logger, use_mnn=True)

            slice1 = batchlist_[0]
            slice2 = batchlist_[1]
            global_PI = np.zeros((len(slice1.obs.index), len(slice2.obs.index)))
            slice1_idx_mapping = {}
            slice2_idx_mapping = {}
            for i in range(len(slice1.obs.index)):
                slice1_idx_mapping[slice1.obs.index[i]] = i
            for i in range(len(slice2.obs.index)):
                slice2_idx_mapping[slice2.obs.index[i]] = i

            # temp_local_pi_list = []
            for i in range(args.num_class):
                subslice1 = slice1[slice1.obs['mclust']==i+1]
                subslice2 = slice2[slice2.obs['mclust']==i+1]
                if subslice1.shape[0]>0 and subslice2.shape[0]>0:
                    pi00 = paste.match_spots_using_spatial_heuristic(subslice1.obsm['spatial'], subslice2.obsm['spatial'], use_ot= True)
                    local_PI = paste.pairwise_align(subslice1, subslice2, alpha=0.1, dissimilarity='kl', use_rep=None, G_init=pi00, use_gpu = True, backend = ot.backend.TorchBackend())
                    for i in range(local_PI.shape[0]):
                        for j in range(local_PI.shape[1]):
                            global_PI[slice1_idx_mapping[subslice1.obs.index[i]]][slice2_idx_mapping[subslice2.obs.index[j]]] = local_PI[i][j]
        else:
             return None
    
    # 
    S = scipy.sparse.csr_matrix(global_PI)
    file = open(os.path.join(exp_fig_dir, "S.pickle"),'wb') #160kb
    pickle.dump(S, file)

    """
    file = open("S.pickle",'rb') 
    S = pickle.load(file)
    S.toarray()

    to retrieve
    """

    # doc aris for boxplot 
    # """write to file for boxplot later"""
    # with open(os.path.join(args.exp_fig_dir, 'MG_ari.txt'), 'a+') as f:
    #     f.write(section_ids[0] + ' ')
    #     f.write(' '.join([str(i) for i in ari_1]))
    #     f.write('\n')
    #     f.write(section_ids[1] + ' ')
    #     f.write(' '.join([str(i) for i in ari_2]))
    #     f.write('\n')
    #     f.write('\n')
    # with open(os.path.join(args.exp_fig_dir, 'MG_ari_pre.txt'), 'a+') as f:
    #     f.write(section_ids[0] + ' ')
    #     f.write(' '.join([str(i) for i in ari_1_pre]))
    #     f.write('\n')
    #     f.write(section_ids[1] + ' ')
    #     f.write(' '.join([str(i) for i in ari_2_pre]))
    #     f.write('\n')
    #     f.write('\n')

    return global_PI, batchlist_


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args_ST()
    # if args.use_cfg:
    #     args = load_best_configs(args)
    # print(args)
    pi, Batch_list = localMG(args)

    slice1 = Batch_list[0]
    slice2 = Batch_list[1]

    layerwise_acc_count =  cal_layer_based_alignment_result(pi, slice1, slice2)

    divisor = slice1.shape[0]
    layerwise_acc_perc = [element/divisor for element in layerwise_acc_count]
    print("the number of spots is:", slice1.shape[0])
    print("LocalOT laywise bar accuracy is:", layerwise_acc_perc)
