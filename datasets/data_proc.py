import logging

import torch

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset

from sklearn.preprocessing import StandardScaler
from .st_loading_utils import load_DLPFC, load_mHypothalamus, load_mMAMP, load_spacelhBC, load_dev_embryo
import scanpy as sc
import pandas as pd
import sklearn.neighbors
import numpy as np
import scipy.sparse as sp
import paste
import ot
import os
import anndata as ad
import scipy
from scipy.sparse import csr_matrix
import networkx as nx


GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset,
}

ST_DICT = {
    "DLPFC", "BC", "mHypothalamus", "mMAMP", "Embryo"
}
        

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, max_neigh=50, model='Radius', verbose=True):
    """
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    # self_loops = pd.DataFrame(zip(Spatial_Net['Cell1'].unique(), Spatial_Net['Cell1'].unique(),
    #                  [0] * len((Spatial_Net['Cell1'].unique())))) ###add self loops
    # self_loops.columns = ['Cell1', 'Cell2', 'Distance']
    # Spatial_Net = pd.concat([Spatial_Net, self_loops], axis=0)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    #########
    try:
        X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    except:
        X = pd.DataFrame(adata.X[:, ], index=adata.obs.index, columns=adata.var.index)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # self-loop
    adata.uns['adj'] = G


def load_ST_dataset(dataset_name, section_ids=["151507", "151508"], hvgs=5000, num_classes=7, is_cached_h5ad=True, st_data_dir="./", h5ad_save_dir="./", is_cached=False, alpha_=0.2, alpha_list=[0.2], ot_pi_root="./"):
    assert dataset_name in ST_DICT, f"Unknow dataset: {dataset_name}."
    name_ = '_'.join(section_ids)
    
    if "DLPFC" in dataset_name:
        # ad_list = []
        if is_cached_h5ad == False:
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
            adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids, uns_merge="same")
            adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
            adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

            adj_concat = np.asarray(adj_list[0].todense())
            for batch_id in range(1,len(section_ids)):
                adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

            """save h5ad of joint slices without inter connections"""
            # adata_concat.X = csr_matrix(adata_concat.X)
            # adj_concat = csr_matrix(adj_concat)

            # adata_concat.uns['adj'] = adj_concat
            # adata_concat.write_h5ad(
            #                 os.path.join(h5ad_save_dir, name_ + '.h5ad')
            #             )
            # adj_concat = adj_concat.toarray()
            # print('adata_concat.shape: ', adata_concat.shape)
        else:
            raise NotImplementedError
            """load h5ad of joint slices without inter connections"""
            adata_concat=sc.read_h5ad(os.path.join(h5ad_save_dir, name_ + '.h5ad'))
            # print(adata_concat.uns)
            print('adata_concat.shape: ', adata_concat.shape)
            adj_concat = adata_concat.uns['adj'].toarray()
            # Batch_list = []
            # adj_list = []
            # for section_id in section_ids:
            #     Batch_list.append(adata_concat[adata_concat.obs['batch_name'] == section_id])
            #     adj_list.append(adata_concat[adata_concat.obs['batch_name'] == section_id].uns['adj'])

        pi_list = []
        pis_list = []
        if is_cached:  
            """with cached inter-connections, we could just load from disc"""
            print("using cached Pis")
            
            for i in range(len(section_ids)-1):
                pi = np.load(os.path.join(ot_pi_root, section_ids[i]+'_'+section_ids[i+1], 'iter0_alpha_'+str(alpha_)+'embedding.npy'))
                pi_list.append(pi)
        else:  
            """if not, let's just calculate it from scratch"""
            print("calculate Pi from scratch")
            temp_ad_list = []
            for section_id in section_ids:
                adata = load_DLPFC(root_dir=st_data_dir, section_id=section_id)
                adata.var_names_make_unique(join="++")
                sc.pp.filter_genes(adata, min_counts = 5)
                temp_ad_list.append(adata)
            for i in range(len(section_ids)-1):
                """for hard encode"""
                if alpha_ != 0:
                    pi0 = paste.match_spots_using_spatial_heuristic(temp_ad_list[i].obsm['spatial'],temp_ad_list[i+1].obsm['spatial'],use_ot=True)
                    pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=alpha_, G_init=pi0, norm=True, numItermax=1000, backend = ot.backend.TorchBackend(), use_gpu=True)
                else:
                    pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=alpha_, G_init=None, norm=True, numItermax=1000, backend = ot.backend.TorchBackend(), use_gpu=True)
                pi_list.append(pi12)
        
            """for otn results"""
            temp_ = []
            for a_ in alpha_list:
                if alpha_ != 0:
                    pi0 = paste.match_spots_using_spatial_heuristic(temp_ad_list[i].obsm['spatial'],temp_ad_list[i+1].obsm['spatial'],use_ot=True)
                    pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=a_, G_init=pi0, norm=True, numItermax=1000, backend = ot.backend.TorchBackend(), use_gpu=True)
                else:
                    pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=a_, G_init=None, norm=True, numItermax=1000, backend = ot.backend.TorchBackend(), use_gpu=True)
                temp_.append(pi12)
            pis_list.append(temp_)

        assert adj_concat.shape[0] == pi_list[0].shape[0] + pi_list[0].shape[1], "adj matrix shape is not consistent with the pi matrix"

        for i in range(pi_list[0].shape[0]):
            for j in range(pi_list[0].shape[1]):
                if pi_list[0][i][j] > 0:
                    adj_concat[i][j+pi_list[0].shape[0]] = 1
                    adj_concat[j+pi_list[0].shape[0]][i] = 1
        
        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph.ndata["feat"] = torch.tensor(adata_concat.X.todense())

    elif "BC" in dataset_name:
        if is_cached_h5ad == False:
            Batch_list = []
            adj_list = []
            for section_id in section_ids:
                ad_ = load_spacelhBC(root_dir=st_data_dir, section_id=section_id)
                ad_.var_names_make_unique(join="++")
            
                # make spot name unique
                ad_.obs_names = [x+'_'+section_id for x in ad_.obs_names]
                
                # Constructing the spatial network
                Cal_Spatial_Net(ad_, rad_cutoff=450) # the spatial network are saved in adata.uns[‘adj’]
                
                # Normalization
                sc.pp.filter_genes(ad_, min_counts = 5)
                sc.pp.highly_variable_genes(ad_, flavor="seurat_v3", n_top_genes=hvgs)
                sc.pp.normalize_total(ad_, target_sum=1e4)
                sc.pp.log1p(ad_)
                ad_ = ad_[:, ad_.var['highly_variable']]

                adj_list.append(ad_.uns['adj'])
                Batch_list.append(ad_)
            adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids, uns_merge="same")
            adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
            adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

            adj_concat = np.asarray(adj_list[0].todense())
            for batch_id in range(1,len(section_ids)):
                adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

            """save h5ad of joint slices without inter connections"""
            # adata_concat.X = csr_matrix(adata_concat.X)
            # adj_concat = csr_matrix(adj_concat)

            # adata_concat.uns['adj'] = adj_concat
            # adata_concat.write_h5ad(
            #                 os.path.join(h5ad_save_dir, name_ + '.h5ad')
            #             )
            # adj_concat = adj_concat.toarray()
            # print('adata_concat.shape: ', adata_concat.shape)
        else:
            """load h5ad of joint slices without inter connections"""
            raise NotImplementedError
            adata_concat=sc.read_h5ad(os.path.join(h5ad_save_dir, name_ + '.h5ad'))
            # print(adata_concat.uns)
            print('adata_concat.shape: ', adata_concat.shape)
            adj_concat = adata_concat.uns['adj'].toarray()
            # Batch_list = []
            # adj_list = []
            # for section_id in section_ids:
            #     Batch_list.append(adata_concat[adata_concat.obs['batch_name'] == section_id])
            #     adj_list.append(adata_concat[adata_concat.obs['batch_name'] == section_id].uns['adj'])

        pi_list = []
        pis_list = []
        if is_cached:  
            """with cached inter-connections, we could just load from disc"""
            print("using cached Pis")
            
            for i in range(len(section_ids)-1):
                pi = np.load(os.path.join(ot_pi_root, section_ids[i]+'_'+section_ids[i+1], 'iter0_alpha_'+str(alpha_)+'embedding.npy'))
                pi_list.append(pi)
        else:  
            """if not, let's just calculate it from scratch"""
            print("calculate Pi from scratch")
            temp_ad_list = []
            for section_id in section_ids:
                adata = load_spacelhBC(root_dir=st_data_dir, section_id=section_id)
                adata.var_names_make_unique(join="++")
                sc.pp.filter_genes(adata, min_counts = 5)
                temp_ad_list.append(adata)
            for i in range(len(section_ids)-1):
                """for hard encode"""
                if alpha_ != 0:
                    pi0 = paste.match_spots_using_spatial_heuristic(temp_ad_list[i].obsm['spatial'],temp_ad_list[i+1].obsm['spatial'],use_ot=True)
                    pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=alpha_, G_init=pi0, norm=True, numItermax=1000, backend = ot.backend.TorchBackend(), use_gpu=True)
                else:
                    pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=alpha_, G_init=None, norm=True, numItermax=1000, backend = ot.backend.TorchBackend(), use_gpu=True)
                pi_list.append(pi12)
        
            """for otn results"""
            temp_ = []
            for a_ in alpha_list:
                if alpha_ != 0:
                    pi0 = paste.match_spots_using_spatial_heuristic(temp_ad_list[i].obsm['spatial'],temp_ad_list[i+1].obsm['spatial'],use_ot=True)
                    pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=a_, G_init=pi0, norm=True, numItermax=1000, backend = ot.backend.TorchBackend(), use_gpu=True)
                else:
                    pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=a_, G_init=None, norm=True, numItermax=1000, backend = ot.backend.TorchBackend(), use_gpu=True)
                temp_.append(pi12)
            pis_list.append(temp_)

        assert adj_concat.shape[0] == pi_list[0].shape[0] + pi_list[0].shape[1], "adj matrix shape is not consistent with the pi matrix"

        for i in range(pi_list[0].shape[0]):
            for j in range(pi_list[0].shape[1]):
                if pi_list[0][i][j] > 0:
                    adj_concat[i][j+pi_list[0].shape[0]] = 1
                    adj_concat[j+pi_list[0].shape[0]][i] = 1
        
        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph.ndata["feat"] = torch.tensor(adata_concat.X.todense())
    elif "mHypothalamus" in dataset_name:
        # ad_list = []
        alpha_=0.1
        alpha_list=[0.1]
        if is_cached_h5ad == False:
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
            adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids, uns_merge="same")
            adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
            adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

            adj_concat = np.asarray(adj_list[0].todense())
            for batch_id in range(1,len(section_ids)):
                adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

            """save h5ad of joint slices without inter connections"""
            # adata_concat.X = csr_matrix(adata_concat.X)
            # adj_concat = csr_matrix(adj_concat)

            # adata_concat.uns['adj'] = adj_concat
            # adata_concat.write_h5ad(
            #                 os.path.join(h5ad_save_dir, name_ + '.h5ad')
            #             )
            # adj_concat = adj_concat.toarray()
            # print('adata_concat.shape: ', adata_concat.shape)
        else:
            """load h5ad of joint slices without inter connections"""
            raise NotImplementedError
            adata_concat=sc.read_h5ad(os.path.join(h5ad_save_dir, name_ + '.h5ad'))
            # print(adata_concat.uns)
            print('adata_concat.shape: ', adata_concat.shape)
            adj_concat = adata_concat.uns['adj'].toarray()
            # Batch_list = []
            # adj_list = []
            # for section_id in section_ids:
            #     Batch_list.append(adata_concat[adata_concat.obs['batch_name'] == section_id])
            #     adj_list.append(adata_concat[adata_concat.obs['batch_name'] == section_id].uns['adj'])

        pi_list = []
        pis_list = []
        if is_cached:  
            """with cached inter-connections, we could just load from disc"""
            print("using cached Pis")
            
            for i in range(len(section_ids)-1):
                pi = np.load(os.path.join(ot_pi_root, section_ids[i]+'_'+section_ids[i+1], 'iter0_alpha_'+str(alpha_)+'embedding.npy'))
                pi_list.append(pi)
        else:  
            """if not, let's just calculate it from scratch"""
            print("calculate Pi from scratch")
            temp_ad_list = []
            for section_id in section_ids:
                adata = load_mHypothalamus(root_dir=st_data_dir, section_id=section_id)
                adata.var_names_make_unique(join="++")
                # sc.pp.filter_genes(adata, min_counts = 5)
                temp_ad_list.append(adata)
            for i in range(len(section_ids)-1):
                """for hard encode"""
                if alpha_ != 0:
                    pi0 = paste.match_spots_using_spatial_heuristic(temp_ad_list[i].obsm['spatial'],temp_ad_list[i+1].obsm['spatial'],use_ot=True)
                    pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=alpha_, G_init=pi0, norm=True, numItermax=2000, backend = ot.backend.TorchBackend(), use_gpu=True)
                else:
                    pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=alpha_, G_init=None, norm=True, numItermax=2000, backend = ot.backend.TorchBackend(), use_gpu=True)
                pi_list.append(pi12)
        
            """for otn results"""
            temp_ = []
            for a_ in alpha_list:
                if alpha_ != 0:
                    pi0 = paste.match_spots_using_spatial_heuristic(temp_ad_list[i].obsm['spatial'],temp_ad_list[i+1].obsm['spatial'],use_ot=True)
                    pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=a_, G_init=pi0, norm=True, numItermax=2000, backend = ot.backend.TorchBackend(), use_gpu=True)
                else:
                    pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=a_, G_init=None, norm=True, numItermax=2000, backend = ot.backend.TorchBackend(), use_gpu=True)
                temp_.append(pi12)
            pis_list.append(temp_)

        assert adj_concat.shape[0] == pi_list[0].shape[0] + pi_list[0].shape[1], "adj matrix shape is not consistent with the pi matrix"

        for i in range(pi_list[0].shape[0]):
            for j in range(pi_list[0].shape[1]):
                if pi_list[0][i][j] > 0:
                    adj_concat[i][j+pi_list[0].shape[0]] = 1
                    adj_concat[j+pi_list[0].shape[0]][i] = 1
        
        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph.ndata["feat"] = torch.tensor(adata_concat.X).float()
    elif "mMAMP" in dataset_name: 
        # ['MA','MP']
        alpha_=0
        alpha_list=[0]
        if is_cached_h5ad == False:
            Batch_list = []
            adj_list = []
            for section_id in section_ids:
                ad_ = load_mMAMP(root_dir=st_data_dir, section_id=section_id)
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
            adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids, uns_merge="same")
            # adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
            adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

            adj_concat = np.asarray(adj_list[0].todense())
            for batch_id in range(1,len(section_ids)):
                adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

            """save h5ad of joint slices without inter connections"""
            # adata_concat.X = csr_matrix(adata_concat.X)
            # adj_concat = csr_matrix(adj_concat)

            # adata_concat.uns['adj'] = adj_concat
            # adata_concat.write_h5ad(
            #                 os.path.join(h5ad_save_dir, name_ + '.h5ad')
            #             )
            # adj_concat = adj_concat.toarray()
            # print('adata_concat.shape: ', adata_concat.shape)
        else:
            """load h5ad of joint slices without inter connections"""
            raise NotImplementedError
            adata_concat=sc.read_h5ad(os.path.join(h5ad_save_dir, name_ + '.h5ad'))
            # print(adata_concat.uns)
            print('adata_concat.shape: ', adata_concat.shape)
            adj_concat = adata_concat.uns['adj'].toarray()
            # Batch_list = []
            # adj_list = []
            # for section_id in section_ids:
            #     Batch_list.append(adata_concat[adata_concat.obs['batch_name'] == section_id])
            #     adj_list.append(adata_concat[adata_concat.obs['batch_name'] == section_id].uns['adj'])

        """get mnn if slices not consecutive"""
        

        pis_list = []
        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph.ndata["feat"] = torch.tensor(adata_concat.X.todense())
    elif "Embryo" in dataset_name:
        # /home/yunfei/spatial_benchmarking/benchmarking_data/Embryo
        alpha_=0
        alpha_list=[0]
        Batch_list = []
        adj_list = []
        embryo_p = st_data_dir
        for section_id in section_ids:
            sec = "E"+section_id+".h5ad"
            ad_ = sc.read_h5ad(os.path.join(embryo_p, sec))
            ad_.var_names_make_unique(join="++")
        
            # make spot name unique
            ad_.obs_names = [x+'_'+section_id for x in ad_.obs_names]
            
            # Constructing the spatial network
            Cal_Spatial_Net(ad_, rad_cutoff=1.3) # the spatial network are saved in adata.uns[‘adj’]
            
            # Normalization
            
            sc.pp.normalize_total(ad_, target_sum=1e4)
            sc.pp.log1p(ad_)

            sc.pp.highly_variable_genes(ad_, flavor="seurat_v3", n_top_genes=hvgs) #ensure enough common HVGs in the combined matrix
            ad_ = ad_[:, ad_.var['highly_variable']]

            adj_list.append(ad_.uns['adj'])
            Batch_list.append(ad_)
        adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids, uns_merge="same")
        # adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
        adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

        adj_concat = np.asarray(adj_list[0].todense())
        for batch_id in range(1,len(section_ids)):
            adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
        
        # pi_list = []
        # pis_list = []
        # if is_cached:  
        #     """with cached inter-connections, we could just load from disc"""
        #     print("using cached Pis")
            
        #     for i in range(len(section_ids)-1):
        #         pi = np.load(os.path.join(ot_pi_root, section_ids[i]+'_'+section_ids[i+1], 'iter0_alpha_'+str(alpha_)+'embedding.npy'))
        #         pi_list.append(pi)
        # else:  
        #     """if not, let's just calculate it from scratch"""
        #     print("calculate Pi from scratch")
        #     temp_ad_list = []
        #     for section_id in section_ids:
        #         sec = "E"+section_id+"_E1S1.h5ad"
        #         adata = sc.read_h5ad(os.path.join(embryo_p, sec))
        #         adata.var_names_make_unique(join="++")
        #         sc.pp.filter_genes(adata, min_counts = 5)
        #         temp_ad_list.append(adata)
        #     for i in range(len(section_ids)-1):
        #         """for hard encode"""
        #         # if alpha_ != 0:
        #         #     # pi0 = paste.match_spots_using_spatial_heuristic(temp_ad_list[i].obsm['spatial'],temp_ad_list[i+1].obsm['spatial'],use_ot=True)
        #         #     pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=alpha_, G_init=None, norm=True, numItermax=1000, backend = ot.backend.TorchBackend(), use_gpu=True)
        #         # else:
        #         pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=alpha_, G_init=None, norm=True, numItermax=1000, backend = ot.backend.TorchBackend(), use_gpu=True)
        #         pi_list.append(pi12)
        
        #     """for otn results"""
        #     temp_ = []
        #     for a_ in alpha_list:
        #         # if alpha_ != 0:
        #         #     pi0 = paste.match_spots_using_spatial_heuristic(temp_ad_list[i].obsm['spatial'],temp_ad_list[i+1].obsm['spatial'],use_ot=True)
        #         #     pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=a_, G_init=pi0, norm=True, numItermax=1000, backend = ot.backend.TorchBackend(), use_gpu=True)
        #         # else:
        #         pi12 = paste.pairwise_align(temp_ad_list[i], temp_ad_list[i+1], alpha=a_, G_init=None, norm=True, numItermax=1000, backend = ot.backend.TorchBackend(), use_gpu=True)
        #         temp_.append(pi12)
        #     pis_list.append(temp_)

        # assert adj_concat.shape[0] == pi_list[0].shape[0] + pi_list[0].shape[1], "adj matrix shape is not consistent with the pi matrix"

        # for i in range(pi_list[0].shape[0]):
        #     for j in range(pi_list[0].shape[1]):
        #         if pi_list[0][i][j] > 0:
        #             adj_concat[i][j+pi_list[0].shape[0]] = 1
        #             adj_concat[j+pi_list[0].shape[0]][i] = 1
        pis_list = []
        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph.ndata["feat"] = torch.tensor(adata_concat.X.todense())
    else:
        # print("not implemented ")
        raise NotImplementedError
    num_features = graph.ndata["feat"].shape[1]
    # num_classes = dataset.num_classes
    return graph, (num_features, num_classes), adata_concat, pis_list


def load_ST_dataset_hard(dataset_name, pi, section_ids=["151507", "151508"], hvgs=5000, st_data_dir="./"):
    assert dataset_name in ST_DICT, f"Unknow dataset: {dataset_name}."
    name_ = '_'.join(section_ids)
    
    if "DLPFC" in dataset_name:
        # ad_list = []
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
        adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids, uns_merge="same")
        adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
        adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

        adj_concat = np.asarray(adj_list[0].todense())
        for batch_id in range(1,len(section_ids)):
            adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

        if pi is not None:
            assert adj_concat.shape[0] == pi.shape[0] + pi.shape[1], "adj matrix shape is not consistent with the pi matrix"

            """keep max"""
            max_values = np.max(pi, axis=1)

            # Create a new array with zero
            pi_keep_argmax = np.zeros_like(pi)

            # Loop through each row and set the maximum value to 1 (or any other desired value)
            for i in range(pi.shape[0]):
                pi_keep_argmax[i, np.argmax(pi[i])] = max_values[i]
            
            pi = pi_keep_argmax
            """"""

            for i in range(pi.shape[0]):
                for j in range(pi.shape[1]):
                    if pi[i][j] > 0:
                        adj_concat[i][j+pi.shape[0]] = 1
                        adj_concat[j+pi.shape[0]][i] = 1
        
        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph.ndata["feat"] = torch.tensor(adata_concat.X.todense())
    elif "mHypothalamus" in dataset_name:
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
        adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids, uns_merge="same")
        adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
        adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

        adj_concat = np.asarray(adj_list[0].todense())
        for batch_id in range(1,len(section_ids)):
            adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

        assert adj_concat.shape[0] == pi.shape[0] + pi.shape[1], "adj matrix shape is not consistent with the pi matrix"

        """keep max"""
        max_values = np.max(pi, axis=1)

        # Create a new array with zero
        pi_keep_argmax = np.zeros_like(pi)

        # Loop through each row and set the maximum value to 1 (or any other desired value)
        for i in range(pi.shape[0]):
            pi_keep_argmax[i, np.argmax(pi[i])] = max_values[i]
        
        pi = pi_keep_argmax
        """"""

        for i in range(pi.shape[0]):
            for j in range(pi.shape[1]):
                if pi[i][j] > 0:
                    adj_concat[i][j+pi.shape[0]] = 1
                    adj_concat[j+pi.shape[0]][i] = 1
        
        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph.ndata["feat"] = torch.tensor(adata_concat.X).float()
    else:
        # print("not implemented ")
        raise NotImplementedError
    num_features = graph.ndata["feat"].shape[1]
    # num_classes = dataset.num_classes
    return graph, num_features, adata_concat
    

def preprocess(graph):
    # make bidirected
    if "feat" in graph.ndata:
        feat = graph.ndata["feat"]
    else:
        feat = None
    src, dst = graph.all_edges()
    # graph.add_edges(dst, src)
    graph = dgl.to_bidirected(graph)
    if feat is not None:
        graph.ndata["feat"] = feat

    # add self-loop
    graph = graph.remove_self_loop().add_self_loop()
    # graph.create_formats_()
    return graph


if __name__ == '__main__':
    g, (num_feats, num_c) = load_ST_dataset(dataset_name="DLPFC", section_ids=["151507", "151508"])
