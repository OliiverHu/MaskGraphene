import scanpy as sc
import os
import pandas as pd
import numpy as np
import anndata
import matplotlib.pyplot as plt
import itertools
import networkx as nx
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import hnswlib
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score as ari_score


plt.rcParams['figure.figsize'] = (9.0, 9.0)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['lines.markersize'] = 10

SMALL_SIZE = 20
MEDIUM_SIZE = 30
BIGGER_SIZE = 35

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['axes.facecolor'] = 'white'


# for loading DLPFC12 data
def load_DLPFC(root_dir='../benchmarking_data/DLPFC12', section_id='151507'):
    # 151507, ..., 151676 12 in total
    ad = sc.read_visium(path=os.path.join(root_dir, section_id), count_file=section_id+'_filtered_feature_bc_matrix.h5', load_images=True)
    ad.var_names_make_unique()

    gt_dir = os.path.join(root_dir, section_id, 'gt')
    gt_df = pd.read_csv(os.path.join(gt_dir, 'tissue_positions_list_GTs.txt'), sep=',', header=None, index_col=0)
    ad.obs['original_clusters'] = gt_df.loc[:, 6]
    keep_bcs = ad.obs.dropna().index
    ad = ad[keep_bcs].copy()
    ad.obs['original_clusters'] = ad.obs['original_clusters'].astype(int).astype(str)
    # print(ad.obs)
    return ad


# for loading BC data
# cluster = 20
def load_BC(root_dir='../benchmarking_data/BC', section_id='section1'):
    # section1
    ad = sc.read_visium(path=os.path.join(root_dir, section_id), count_file=section_id+'_filtered_feature_bc_matrix.h5', load_images=True)
    ad.var_names_make_unique()
    
    gt_dir = os.path.join(root_dir, section_id, 'gt')
    gt_df = pd.read_csv(os.path.join(gt_dir, 'tissue_positions_list_GTs.txt'), sep=',', header=None, index_col=0)
    ad.obs['original_clusters'] = gt_df.loc[:, 6].astype(int)
    ad.obs['original_clusters'] += 1
    keep_bcs = ad.obs.dropna().index
    ad = ad[keep_bcs].copy()
    ad.obs['original_clusters'] = ad.obs['original_clusters'].astype(int).astype(str)
    return ad


# for loading mouse_PFC data
def load_mPFC(root_dir = '/home/yunfei/spatial_benchmarking/benchmarking_data/STARmap_mouse_PFC', section_id='20180417_BZ5_control'):  
    # section id = '20180417_BZ5_control', '20180419_BZ9_control', '20180424_BZ14_control' 3 in total
    # cluster       4                       4                       4
    info_file = os.path.join(root_dir, 'starmap_mpfc_starmap_info.xlsx')
    cnts_file = os.path.join(root_dir, 'starmap_mpfc_starmap_cnts.xlsx')
    
    xls_cnts = pd.ExcelFile(cnts_file)
    df_cnts = pd.read_excel(xls_cnts, section_id)
    
    xls_info = pd.ExcelFile(info_file)
    df_info = pd.read_excel(xls_info, section_id)

    spatial_X = df_info.to_numpy()
    obs_ = df_info
    obs_.columns = ['psuedo_barcodes', 'x', 'y', 'gt', 'original_clusters']
    obs_.index = obs_['psuedo_barcodes'].tolist()
    
    var_ = df_cnts.iloc[:, 0]
    var_ = pd.DataFrame(var_)
    
    ad = anndata.AnnData(X=df_cnts.iloc[:,1:].T, obs=obs_, var=var_)
    ad.obs['original_clusters'] = ad.obs['original_clusters'].astype(int).astype(str)
    spatial = np.vstack((ad.obs['x'].to_numpy(), ad.obs['y'].to_numpy()))
    ad.obsm['spatial'] = spatial.T
    return ad


# for loading mouse_visual_cortex data
# cluster = 7
def load_mVC(root_dir='../benchmarking_data/STARmap_mouse_visual_cortex', section_id='STARmap_20180505_BY3_1k.h5ad'):
    ad = sc.read(os.path.join(root_dir, section_id))
    ad.var_names_make_unique()
    ad.obs.columns = ['Total_counts', 'imagerow', 'imagecol', 'original_clusters']
    return ad


# for loading mHypothalamus data
# already preprocessed? Xs are floats
def load_mHypothalamus(root_dir='/home/yunfei/spatial_benchmarking/benchmarking_data/mHypothalamus', section_id='0.26'):
    # section id = '0.26', '0.21', '0.16', '0.11', '0.06', '0.01', '-0.04', '-0.09', '-0.14', '-0.19', '-0.24', '-0.29' 12 in total
    # cluster =     15      15      14      15      15      15      14       15       15       15      16        15
    info_file = os.path.join(root_dir, 'MERFISH_Animal1_info.xlsx')
    cnts_file = os.path.join(root_dir, 'MERFISH_Animal1_cnts.xlsx')
    xls_cnts = pd.ExcelFile(cnts_file)
    # print(xls_cnts.sheet_names)
    df_cnts = pd.read_excel(xls_cnts, section_id)
    
    xls_info = pd.ExcelFile(info_file)
    df_info = pd.read_excel(xls_info, section_id)
    # print(df_cnts, df_info)
    spatial_X = df_info.to_numpy()
    obs_ = df_info
    if len(df_info.columns) == 5:
        obs_.columns = ['psuedo_barcodes', 'x', 'y', 'original_clusters', 'Neuron_cluster_ID']
    elif len(df_info.columns) == 6:
        obs_.columns = ['psuedo_barcodes', 'x', 'y', 'cell_types', 'Neuron_cluster_ID', 'original_clusters']
        # print(section_id)
        # print(obs_['z'].nunique())
    obs_.index = obs_['psuedo_barcodes'].tolist()
    # print(obs_)

    var_ = df_cnts.iloc[:, 0]
    var_ = pd.DataFrame(var_)
    # print(var_)
    
    ad = anndata.AnnData(X=df_cnts.iloc[:,1:].T, obs=obs_, var=var_)
    spatial = np.vstack((ad.obs['x'].to_numpy(), ad.obs['y'].to_numpy()))
    ad.obsm['spatial'] = spatial.T
    return ad

# for loading her2_tumor data
def load_her2_tumor(root_dir='../benchmarking_data/Her2_tumor', section_id='A1'):
    # section id = A1(348) B1(295) C1(177) D1(309) E1(587) F1(692) G2(475) H1(613) ~J1(254), 8 in total
    # clusters =   6       5       4       4       4       4       7       7
    cnts_dir = os.path.join(root_dir, 'ST-cnts')
    gt_dir = os.path.join(root_dir, 'ST-pat/lbl')
    gt_file_name = section_id+'_labeled_coordinates.tsv'
    cnt_file_name = section_id+'.tsv'
    cnt_df = pd.read_csv(os.path.join(cnts_dir, cnt_file_name), sep='\t', header=0)
    # print(cnt_file_name)
    # print(cnt_df)
    gt_df = pd.read_csv(os.path.join(gt_dir, gt_file_name), sep='\t', header=0)
    # print(gt_file_name)
    # print(gt_df)
    keep_bcs = gt_df.dropna().index
    gt_df = gt_df.iloc[keep_bcs]
    xs = gt_df['x'].tolist()
    ys = gt_df['y'].tolist()
    # print(xs)
    # print(ys)
    rounded_xs = [round(elem) for elem in xs]
    # print(rounded_xs)
    rounded_ys = [round(elem) for elem in ys]
    # print(rounded_ys)

    res = [str(i) + 'x' + str(j) for i, j in zip(rounded_xs, rounded_ys)]
    # print(len(set(res)))
    gt_df['Row.names'] = res
    # print(gt_df)

    spatial_X = cnt_df.to_numpy()
    obs_ = gt_df
    obs_ = obs_.sort_values(by=['Row.names'])
    obs_ = obs_.loc[obs_['Row.names'].isin(cnt_df['Unnamed: 0'])]
    obs_ = obs_.reset_index(drop=True)
    # print(obs_)

    var_ = cnt_df.iloc[0, 1:]
    var_ = pd.DataFrame(var_)

    ad = anndata.AnnData(X=cnt_df.iloc[:,1:], obs=obs_, var=var_, dtype=np.int64)
    ad.obs['original_clusters'] = ad.obs['label']
    spatial = np.vstack((ad.obs['pixel_x'].to_numpy(), ad.obs['pixel_y'].to_numpy()))
    ad.obsm['spatial'] = spatial.T
    return ad


# 
# cluster = 52
def load_mMAMP(root_dir='/home/yunfei/spatial_benchmarking/benchmarking_data/mMAMP', section_id='MA'):
    # if section_id == "MA":
    #     ad = sc.read_visium(path=os.path.join(root_dir, section_id), count_file=section_id+'_filtered_feature_bc_matrix.h5', load_images=True)
    #     ad.var_names_make_unique()
    # elif section_id == "MP":
    fn = "/home/yunfei/spatial_benchmarking/benchmarking_data/mMAMP/" + section_id + "/" + section_id + "1.h5ad"
    # print(fn)
    ad = sc.read_h5ad(filename=fn)
    ad.var_names_make_unique()
    spatial = np.vstack((ad.obs['x4'].to_numpy(), ad.obs['x5'].to_numpy()))
    ad.obsm['spatial'] = spatial.T

    # try:
    #     gt_dir = os.path.join(root_dir, section_id, 'gt')
    #     gt_df = pd.read_csv(os.path.join(gt_dir, 'tissue_positions_list_GTs.txt'), sep='\t', header=0, index_col=0)
    #     ad.obs = gt_df
    #     ad.obs['original_clusters'] = ad.obs['ground_truth']
    # except:
    #     pass
    
    # keep_bcs = ad.obs.dropna().index
    # ad = ad[keep_bcs].copy()
    return ad


# cluster = 10,  
def load_spacelhBC(root_dir='/home/yunfei/spatial_benchmarking/benchmarking_data/visium_human_breast_cancer', section_id='human_bc_spatial_1142243F'):
    # sectionid = filename in this dataset
    # filename [human_bc_spatial_1142243F, human_bc_spatial_1160920F, human_bc_spatial_CID4290, human_bc_spatial_CID4465, human_bc_spatial_CID4535, human_bc_spatial_CID44971, human_bc_spatial_Parent_Visium_Human_BreastCancer]
    # filename [human_bc_spatial_V1_Breast_Cancer_Block_A_Section_1, human_bc_spatial_V1_Breast_Cancer_Block_A_Section_2, human_bc_spatial_V1_Human_Invasive_Ductal_Carcinoma, human_bc_spatial_Visium_FFPE_Human_Breast_Cancer]
    ad = sc.read_h5ad(filename=os.path.join(root_dir, section_id + '.h5ad'))
    ad.var_names_make_unique()
    # print(ad.obs)

    ad.obs['original_clusters'] = ad.obs['spatial_domain']
    return ad


def load_dev_embryo():
    return None


def otn(ot_pi, names1, names2, conf_thres, mode='normal'):
    """
    mode = normal / argmax
    """
    # print(ot_pis)
    # print(ot_pi)
    # print(len(names1))
    # print(len(names2))
    # print(ot_pis.shape)
    # print(len(ot_pis))
    # for ot_pi in [ot_pis]:
    #     print(ot_pi.shape)
    assert ot_pi.shape[0] == len(names1) and ot_pi.shape[1] == len(names2), "mapping matrix does not align with the given name lists"
    # if mode == 'argmax':
    #     print("using pi in argmax mode")
    #     # Find the maximum value in each row
    #     max_values = np.max(ot_pi, axis=1)

    #     # Create a new array with zero
    #     ot_pi_keep_argmax = np.zeros_like(ot_pi)

    #     # Loop through each row and set the maximum value to 1 (or any other desired value)
    #     for i in range(ot_pi.shape[0]):
    #         ot_pi_keep_argmax[i, np.argmax(ot_pi[i])] = max_values[i]
    # else:
    #     print("using pi in normal mode")
    
    # print(ot_pi.shape)
    # print(len(names1), len(names2))
    

    mutual = set()
    # k = 0
    # print(np.count_nonzero(ot_pi_keep_argmax))
    # for ot_pi in ot_pis:
    for i in range(len(names1)):
        for j in range(len(names2)):
            if ot_pi[i][j] > conf_thres:
                # k += 1
                # print(ot_pi[i][j])
                mutual.add((names1[i], names2[j]))
    # print(k)
    return mutual


def mnn(ds1, ds2, names1, names2, knn = 20, save_on_disk = True, approx = True):
    if approx: 
        # Find nearest neighbors in first direction.
        # output KNN point for each point in ds1.  match1 is a set(): (points in names1, points in names2), the size of the set is ds1.shape[0]*knn
        match1 = nn_approx(ds1, ds2, names1, names2, knn=knn)#, save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.
        match2 = nn_approx(ds2, ds1, names2, names1, knn=knn)#, save_on_disk = save_on_disk)
    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn)
        match2 = nn(ds2, ds1, names2, names1, knn=knn)
    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual


def nn_approx(ds1, ds2, names1, names2, knn=50):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M = 16)
    p.set_ef(10)
    p.add_items(ds2)
    ind,  distances = p.knn_query(ds1, k=knn)
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))
    return match


def nn(ds1, ds2, names1, names2, knn=50, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


"""OT pi as inter-slice neighbors"""
def create_dictionary_otn(adata, pis, section_ids, batch_name, conf_thres = 0.0, mode='normal', verbose = 1, iter_comb = None):

    cell_names = adata.obs_names

    batch_list = adata.obs[batch_name]
    datasets = []
    # datasets_pcs = []
    cells = []
    for i in batch_list.unique():
        datasets.append(adata[batch_list == i])
        # datasets_pcs.append(adata[batch_list == i].obsm[use_rep])
        cells.append(cell_names[batch_list == i])

    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
    otns = dict()

    if iter_comb is None:
        iter_comb = list(itertools.combinations(range(len(cells)), 2))

    # print(iter_comb)
    assert len(iter_comb) == len(pis)
    id_ = 0
    for comb in iter_comb:
        i = comb[0]
        j = comb[1]
        key_name1 = batch_name_df.loc[comb[0]].values[0] + "_" + batch_name_df.loc[comb[1]].values[0]
        otns[key_name1] = {} # for multiple-slice setting, the key_names1 can avoid the mnns replaced by previous slice-pair
        if(verbose > 0):
            print('Processing datasets {}'.format((i, j)))

        new = list(cells[j])
        ref = list(cells[i])
        # print(len(new))
        # print(len(ref))
        # print(pis[id_].shape)

        # ds1 = adata[new].obsm[use_rep]
        # ds2 = adata[ref].obsm[use_rep]
        ot_pi = pis[id_]
        id_ += 1
        # names1 = new
        # names2 = ref
        
        """mapping direction might be a issue here"""
        # mapping: new -> ref (151674[3635] -> 151673[3611])
        # pi ()
        #          ref: bc1, bc2, ..., bcn
        #   new:    1   0    1    ...  0
        #   bc1,    0   0    1    ...  1
        #   bc2,    1   0    0    ...  0
        #   ...,       ...
        #   bcn     1   0    0    ...  1
        # ot_pi is the ot transport matrix
        # names2 are the ref barcodes
        # names1 are the new barcodes
        match = otn(ot_pi, ref, new, conf_thres, mode=mode)
        # print(type(match))
        # print(len(match))
        # exit(-1)
        G = nx.Graph()
        G.add_edges_from(match)
        node_names = np.array(G.nodes)
        anchors = list(node_names)
        adj = nx.adjacency_matrix(G)
        tmp = np.split(adj.indices, adj.indptr[1:-1])

        for i in range(0, len(anchors)):
            key = anchors[i]
            i = tmp[i]
            names = list(node_names[i])
            otns[key_name1][key]= names
    return(otns)


"""mnn as inter-slice neighbors"""
def create_dictionary_mnn(adata, use_rep, batch_name, k = 150, save_on_disk = True, approx = True, verbose = 1, iter_comb = None):

    cell_names = adata.obs_names  # obs index (barcodes) of all spots

    batch_list = adata.obs[batch_name]
    datasets = []
    datasets_pcs = []
    cells = []  # a list of indexes of each slice separately
    for i in batch_list.unique():
        datasets.append(adata[batch_list == i])
        datasets_pcs.append(adata[batch_list == i].obsm[use_rep])
        cells.append(cell_names[batch_list == i])

    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
    # print(batch_name_df)
    mnns = dict()

    if iter_comb is None:  # pairs of slices
        iter_comb = list(itertools.combinations(range(len(cells)), 2))
    
    # print("cell names")
    # print(cell_names)
    # print("cells")
    # print(cells)
    # print(iter_comb[:20])
    for comb in iter_comb:
        i = comb[0]
        j = comb[1]
        key_name1 = batch_name_df.loc[comb[0]].values[0] + "_" + batch_name_df.loc[comb[1]].values[0]
        mnns[key_name1] = {} # for multiple-slice setting, the key_names1 can avoid the mnns replaced by previous slice-pair
        if(verbose > 0):
            print('Processing datasets {}'.format((i, j)))
        
        # mapping: new -> ref (151674 -> 151673)
        new = list(cells[j])
        ref = list(cells[i])
        # print(new)
        # print(ref)

        ds1 = adata[new].obsm[use_rep]
        ds2 = adata[ref].obsm[use_rep]
        names1 = new
        names2 = ref
        # if k>1，one point in ds1 may have multiple MNN points in ds2.
        match = mnn(ds1, ds2, names1, names2, knn=k, save_on_disk = save_on_disk, approx = approx)
        # print(type(match))
        # print(len(match))
        # print(match[:10])
        # exit(-1)

        G = nx.Graph()
        G.add_edges_from(match)
        node_names = np.array(G.nodes)
        anchors = list(node_names)
        adj = nx.adjacency_matrix(G)
        tmp = np.split(adj.indices, adj.indptr[1:-1])

        for i in range(0, len(anchors)):
            key = anchors[i]
            i = tmp[i]
            names = list(node_names[i])
            mnns[key_name1][key]= names
    return(mnns)


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='MDOT', random_seed=666):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    print(adata.obsm[used_obsm])
    res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)

    # print(res)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    # adata.obs['mclust'] = adata.obs['mclust'].astype('string')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def gmm_scikit(adata, num_cluster, modelNames='gmm', used_obsm='MDOT', random_seed=666):
    X = adata.obsm[used_obsm]
    bgm = GaussianMixture(n_components=num_cluster, random_state=random_seed).fit(X)
    # bgm.means_
    lbls_gmm = bgm.predict(X)

    adata.obs['mclust'] = lbls_gmm
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    # adata.obs['mclust'] = adata.obs['mclust'].astype('string')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


# cluster = 7/5  
def load_dlpfc_sim(root_dir='./', section_id=''):
    ad = sc.read_h5ad(filename=os.path.join(root_dir, section_id + '.h5ad'))
    ad.var_names_make_unique()

    ad.obs['original_clusters'] = ad.obs['spatial_domain']
    return ad


# visualize everything for sanity check
def anndata_visualization(ad, fname, save_folder='/home/yunfei/spatial_benchmarking/benchmarking_data/gt_visualization', col_name='original_clusters', spot_size=150):
    sc.pl.spatial(ad,
                  color=[col_name],
                  title=[col_name],
                  show=True, spot_size=spot_size)
    plt.savefig(os.path.join(save_folder, fname + ".pdf"))


def visualization_umap_spatial(ad_temp, section_ids, exp_fig_dir, dataset_name, num_iter, identifier, num_class, use_key=""):
    i = num_iter

    mclust_R(ad_temp, modelNames='EEE', num_cluster=num_class, used_obsm=use_key)

    """umap"""
    sc.pp.neighbors(ad_temp, use_rep=use_key, random_state=666)
    # print(ad_concat.obsp["connectivities"])
    sc.tl.umap(ad_temp, random_state=666)

    section_color = ['#f8766d', '#7cae00']
    section_color_dict = dict(zip(section_ids, section_color))
    ad_temp.uns['batch_name_colors'] = [section_color_dict[x] for x in ad_temp.obs.batch_name.cat.categories]

    fig = sc.pl.umap(ad_temp, color=['batch_name', 'original_clusters', 'mclust'], ncols=3, wspace=0.5,return_fig=True)  # , save=os.path.join('/dot/0914/', dataset+'_umap.pdf')
    fig.savefig(os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids)+str(i)+identifier+".pdf"))

    """visualization"""
    if dataset_name == "mHypothalamus":
        spot_size=25
        title_size = 12
    elif dataset_name == "DLPFC":
        spot_size=75
        title_size = 12
    elif dataset_name == "BC":
        spot_size=175
        title_size = 12
    else:
        raise NotImplementedError
    
    Batch_list = []
    for section_id in section_ids:
        Batch_list.append(ad_temp[ad_temp.obs['batch_name'] == section_id])
    
    ARI_list = []
    # ari_1_pre.append(round(ari_score(Batch_list[0].obs['original_clusters'], Batch_list[0].obs['mclust']), 3))
    # ari_2_pre.append(round(ari_score(Batch_list[1].obs['original_clusters'], Batch_list[1].obs['mclust']), 3))
    ARI_list.append(round(ari_score(Batch_list[0].obs['original_clusters'], Batch_list[0].obs['mclust']), 3))
    ARI_list.append(round(ari_score(Batch_list[1].obs['original_clusters'], Batch_list[1].obs['mclust']), 3))

    fig, ax = plt.subplots(1, 4, figsize=(20, 10)) # gridspec_kw={'wspace': 0.05, 'hspace': 0.1}
    _sc_0 = sc.pl.spatial(Batch_list[0], img_key=None, color=['mclust'], title=[''],
                        legend_fontsize=9, show=False, ax=ax[0], frameon=False,
                        spot_size=spot_size)
    _sc_0[0].set_title("ARI=" + str(ARI_list[0]), size=title_size)
    _sc_0_gt = sc.pl.spatial(Batch_list[0], img_key=None, color=['original_clusters'], title=[''],
                        legend_fontsize=9, show=False, ax=ax[1], frameon=False,
                        spot_size=spot_size)
    _sc_0_gt[0].set_title("Ground Truth", size=title_size)
    _sc_1 = sc.pl.spatial(Batch_list[1], img_key=None, color=['mclust'], title=[''],
                        legend_fontsize=9, show=False, ax=ax[2], frameon=False,
                        spot_size=spot_size)
    _sc_1[0].set_title("ARI=" + str(ARI_list[1]), size=title_size)
    _sc_1_gt = sc.pl.spatial(Batch_list[1], img_key=None, color=['original_clusters'], title=[''],
                        legend_fontsize=9, show=False, ax=ax[3], frameon=False,
                        spot_size=spot_size)
    _sc_1_gt[0].set_title("Ground Truth", size=title_size)
    plt.savefig(os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids)+str(i)+"viz_" +identifier+ ".pdf"))
    # plt.show()
    
    """save embedding + labels"""
    df_labels = ad_temp.obs[['original_clusters', 'mclust']]
    embed_ = ad_temp.obsm[use_key]
    np.save(os.path.join(exp_fig_dir, '_'.join(section_ids)+'iter'+str(i)+'embedding_'+identifier+'.npy'), embed_)
    df_labels.to_csv(os.path.join(exp_fig_dir, '_'.join(section_ids)+'iter'+str(i)+'labels_'+identifier+'.csv'))

    return ARI_list


def cal_layer_based_alignment_result(alignment, s1, s2):
    #fei added
    labels = []
    labels.extend(s1.obs['original_clusters'])
    labels.extend(s2.obs['original_clusters'])

    total = alignment.shape[0]
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
    
    res.extend([cnt0/total, cnt1/total, cnt2/total, cnt3/total, cnt4/total, cnt5/total, cnt6/total])
    return res


if __name__ == '__main__':
    # for sec in ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']:
    #     ad = load_DLPFC(root_dir='/home/yunfei/spatial_benchmarking/benchmarking_data/DLPFC12', section_id=sec)
    #     # print(ad.obs)
    #     # exit(-1)
    #     # spatial = np.vstack((ad.obs['pixel_x'].to_numpy(), ad.obs['pixel_y'].to_numpy()))
    #     # ad.obsm['spatial'] = spatial.T
    #     # anndata_visualization(ad, 'dlpfc_' + sec, col_name='Ground Truth', spot_size=75)
    #     print(sec)
    #     print(ad.X.shape)
    # print("dlpfc test passed")

    # ad = load_BC(root_dir='/home/yunfei/spatial_benchmarking/benchmarking_data/BC', section_id='section1')
    # # anndata_visualization(ad, 'bc_section1', col_name='Ground Truth', spot_size=155)
    # # print(sec)
    # print(ad.X.shape)
    # print("bc test passed")

    # for sec in ['20180417_BZ5_control', '20180419_BZ9_control', '20180424_BZ14_control']:
    #     ad = load_mPFC(root_dir = '/home/yunfei/spatial_benchmarking/benchmarking_data/STARmap_mouse_PFC', section_id=sec)
    #     spatial = np.vstack((ad.obs['x'].to_numpy(), ad.obs['y'].to_numpy()))
    #     ad.obsm['spatial'] = spatial.T
    #     # print(ad.obs)
    #     # anndata_visualization(ad, 'mpfc_' + sec, col_name='z', spot_size=175)
    #     print(sec)
    #     print(ad.X.shape)
    # print("mPFC test passed")

    # for sec in ['-0.04', '-0.09', '-0.14', '-0.19', '-0.24', '-0.29']:
    #     ad = load_mHypothalamus(root_dir='/home/yunfei/spatial_benchmarking/benchmarking_data/mHypothalamus', section_id=sec)
    #     print(ad.obs)
    #     spatial = np.vstack((ad.obs['x'].to_numpy(), ad.obs['y'].to_numpy()))
    #     ad.obsm['spatial'] = spatial.T
    #     print(sec)
    #     print(ad.X.shape)
    #     # exit(-1)
    #     # anndata_visualization(ad, 'mHypo_' + sec, col_name='GT', spot_size=30)
    #     # print(sec)
    #     # print(ad.obs['GT'].nunique())
    # print("mH test passed")

    # ad = load_mVC(root_dir='/home/yunfei/spatial_benchmarking/benchmarking_data/STARmap_mouse_visual_cortex', section_id='STARmap_20180505_BY3_1k.h5ad')
    # print(ad.obs)
    # # anndata_visualization(ad, 'mvc_' + 'STARmap_20180505_BY3_1k.h5ad', col_name='Ground Truth', spot_size=175)
    # # print(sec)
    # print(ad.X.shape)
    # print("mVC test passed")

    # for sec in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']:
    #     ad = load_her2_tumor(root_dir='/home/yunfei/spatial_benchmarking/benchmarking_data/Her2_tumor', section_id=sec)
    #     spatial = np.vstack((ad.obs['pixel_x'].to_numpy(), ad.obs['pixel_y'].to_numpy()))
    #     ad.obsm['spatial'] = spatial.T
    #     # anndata_visualization(ad, 'her2tumor_' + sec, col_name='label', spot_size=75)
    #     print(sec)
    #     print(ad.X.shape)
    # print("her2tumor test passed")

    # ad = load_mMAMP()
    # # print(ad.obs)
    # # anndata_visualization(ad, 'mMAMP_anterior', col_name='ground_truth', spot_size=80)
    # print(ad.X.shape)

    for sec in ['human_bc_spatial_1142243F', 'human_bc_spatial_1160920F', 'human_bc_spatial_CID4290', 'human_bc_spatial_CID4465', 'human_bc_spatial_CID4535', 'human_bc_spatial_CID44971', 'human_bc_spatial_Parent_Visium_Human_BreastCancer',
                'human_bc_spatial_V1_Breast_Cancer_Block_A_Section_1', 'human_bc_spatial_V1_Breast_Cancer_Block_A_Section_2', 'human_bc_spatial_V1_Human_Invasive_Ductal_Carcinoma', 'human_bc_spatial_Visium_FFPE_Human_Breast_Cancer']:
        ad = load_spacelhBC(root_dir='/home/yunfei/spatial_benchmarking/benchmarking_data/visium_human_breast_cancer', section_id=sec)
        # spatial = np.vstack((ad.obs['pixel_x'].to_numpy(), ad.obs['pixel_y'].to_numpy()))
        # ad.obsm['spatial'] = spatial.T
        anndata_visualization(ad, sec, spot_size=75)
        print(sec)
        print(ad.X.shape)
    print("spacelhBC test passed")


