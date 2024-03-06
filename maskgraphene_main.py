import logging
import numpy as np
from tqdm import tqdm
import torch
import pickle
import wandb

from utils import (
    build_args_ST,
    create_optimizer,
    set_random_seed,
    get_current_lr,
)
from datasets.data_proc import load_ST_dataset_hard, load_ST_dataset_hard_erase
from datasets.st_loading_utils import create_dictionary_otn, visualization_umap_spatial, create_dictionary_mnn, cal_layer_based_alignment_result, cal_layer_based_alignment_result_mhypo
from models import build_model_ST
import os
import scanpy as sc
import sklearn.metrics.pairwise


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


# def train_MDOT(model, feats, graph, max_epoch, device, use_scheduler, lr, weight_decay, batch_size=512, sampling_method="lc", optimizer="adam", drop_edge_rate=0):
def MG(model, graph, feat, optimizer, max_epoch, device, adata_concat_, scheduler, key_="MG", logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    target_nodes = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()
        loss = model(graph, x, targets=target_nodes)

        loss_dict = {"loss": loss.item()}
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    with torch.no_grad():
        z = model.embed(graph, x)
        # print(z)
    
    # model.eval()
    adata_concat_.obsm[key_] = z.cpu().detach().numpy()
    
    return model, adata_concat_


def MG_triplet(model, graph, feat, optimizer, max_epoch, device, adata_concat_, pis_list_, scheduler, logger=None, key_="MG_triplet", iter_comb=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    target_nodes = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
    epoch_iter = tqdm(range(max_epoch))
    section_ids = np.array(adata_concat_.obs['batch_name'].unique())

    """a list of precomputed pi, it has the same length as the iter_comb"""
    # mnn_dict = create_dictionary_otn(adata_concat_, pis_list_, section_ids, batch_name='batch_name', conf_thres = 0.0, mode="normal", verbose = 1, iter_comb=iter_comb)
    mnn_dict = create_dictionary_mnn(adata_concat_, use_rep="MG", batch_name='batch_name', k=50, iter_comb=iter_comb)
    anchor_ind = []
    positive_ind = []
    negative_ind = []
    for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
        batchname_list = adata_concat_.obs['batch_name'][mnn_dict[batch_pair].keys()]
        #             print("before add KNN pairs, len(mnn_dict[batch_pair]):",
        #                   sum(adata_new.obs['batch_name'].isin(batchname_list.unique())), len(mnn_dict[batch_pair]))

        cellname_by_batch_dict = dict()
        for batch_id in range(len(section_ids)):
            cellname_by_batch_dict[section_ids[batch_id]] = adata_concat_.obs_names[
                adata_concat_.obs['batch_name'] == section_ids[batch_id]].values

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

        batch_as_dict = dict(zip(list(adata_concat_.obs_names), range(0, adata_concat_.shape[0])))
        anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
        positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
        negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

    for epoch in epoch_iter:


        model.train()
        optimizer.zero_grad()

        _loss = model(graph, x, targets=target_nodes)
        with torch.no_grad():
            z = model.embed(graph, x)

        anchor_arr = z[anchor_ind,]
        positive_arr = z[positive_ind,]
        negative_arr = z[negative_ind,]

        triplet_loss = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')
        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

        loss = _loss + tri_output
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        # loss_dict = {"loss": loss.item()}
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        
        if logger is not None:
            # loss_dict["lr"] = get_current_lr(optimizer)
            # logger.log(loss_dict, step=epoch)
            logger.log({'epoch': epoch+1, 'ratio': r, 'layer-wise acc': lw_acc[0], 'loss':loss.item()})

    with torch.no_grad():
        z = model.embed(graph, x)
    
    # model.eval()
    adata_concat_.obsm[key_] = z.cpu().detach().numpy()

    return model, adata_concat_


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_triplet = args.max_epoch_triplet

    # num_hidden = args.num_hidden
    # num_layers = args.num_layers
    # encoder_type = args.encoder
    # decoder_type = args.decoder
    # replace_rate = args.replace_rate
    is_consecutive = args.consecutive_prior

    optim_type = args.optimizer 
    # loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    logs = args.logging
    logs = False
    use_scheduler = args.scheduler

    """ST loading"""
    section_ids = args.section_ids.lstrip().split(",")
    
    """file save path"""
    exp_fig_dir = args.exp_fig_dir
    st_data_dir = args.st_data_dir
    pi_dir= os.path.join(args.pi_dir, dataset_name+'_'.join(section_ids))

    model_dir = "checkpoints"
    os.makedirs(model_dir, exist_ok=True)
    # print(logs)
    

    # acc_list = []
    # estp_acc_list = []
    ari_1 = []
    ari_2 = []
    ari_1_pre = []
    ari_2_pre = []
    if not os.path.exists(os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids))):
        os.makedirs(os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids)))
    # exp_fig_dir = os.path.join("/home/yunfei/spatial_dl_integration/MaskGraphene/exps_1016_2", dataset_name+'_'.join(section_ids))
    # if dataset_name == "DLPFC":
    #     pi_dir = os.path.join("../PI", dataset_name+'_'.join(section_ids))
    # if dataset_name == "mHypothalamus":
    #     pi_dir = os.path.join("../PI", dataset_name+'_'.join(section_ids))
    exp_fig_dir = os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids))

    counter = 0
    # use_cached_pi = True
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            # logger = WandbLogger(log_path=f"{dataset_name}_{'_'.join(section_ids)}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}__wd_{weight_decay}__{encoder_type}_{decoder_type}", project="M-DOT", args=args)
            logger = wandb.init(name='_'.join(section_ids)+"_seed"+str(seed))
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
        
        if is_consecutive:
            file = open(os.path.join(pi_dir, "S.pickle"),'rb') 
            global_PI = pickle.load(file)
            global_PI = global_PI.toarray()
            """
            STAGE 1
            """
            
            """train with MSSL"""
            graph, num_features, ad_concat = load_ST_dataset_hard_erase(dataset_name=dataset_name, pi=global_PI, section_ids=section_ids, hvgs=args.hvgs, st_data_dir=st_data_dir)
            args.num_features = num_features
            x = graph.ndata["feat"]

            print(args)
            model = build_model_ST(args)
            print(model)
            model.to(device)
            optimizer = create_optimizer(optim_type, model, lr, weight_decay)

            if use_scheduler:
                logging.info("Use scheduler")
                scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
            else:
                scheduler = None

            model.to(device)
            graph = graph.to(device)
            x = x.to(device)
            # print(ad_concat[0])
            model, ad_concat_1 = MG(model, graph, x, optimizer, max_epoch, device, ad_concat, scheduler, logger=logger, key_="MG")
            # print(ad_concat_1)
            # print(ad_concat_1.obsm["MG"])
            ari_ = visualization_umap_spatial(ad_temp=ad_concat_1, section_ids=section_ids, exp_fig_dir=exp_fig_dir, dataset_name=dataset_name, num_iter=counter, identifier="stage1", num_class=args.num_class, use_key="MG")
            ari_1_pre.append(ari_[0])
            ari_2_pre.append(ari_[1])
            if logger is not None:
                logger.log({"slice1_ari_pre": ari_[0], "slice2_ari_pre": ari_[1]})
            # print(section_id)
            print(section_ids[0], ', ARI = %01.3f' % ari_[0])
            print(section_ids[1], ', ARI = %01.3f' % ari_[1])
            # exit(-1)
            """train with MSSL + triplet loss"""
            logging.info("Keep training Model with cse + triplet loss ")

            model, ad_concat_2 = MG_triplet(model, graph, x, optimizer, max_epoch_triplet, device, adata_concat_=ad_concat_1, pis_list_=[global_PI], scheduler=scheduler, logger=logger, key_="MG_triplet")
            ari_ = visualization_umap_spatial(ad_temp=ad_concat_2, section_ids=section_ids, exp_fig_dir=exp_fig_dir, dataset_name=dataset_name, num_iter=counter, identifier="stage2", num_class=args.num_class, use_key="MG_triplet")
            counter += 1
            ari_1.append(ari_[0])
            ari_2.append(ari_[1])
            if logger is not None:
                logger.log({"slice1_ari_after": ari_[0], "slice2_ari_after": ari_[1]})
            print(section_ids[0], ', ARI = %01.3f' % ari_[0])
            print(section_ids[1], ', ARI = %01.3f' % ari_[1])
        else:
            global_PI = None
            """
            STAGE 2
            """

            """train with MSSL"""
            logging.info("training Model with sce + triplet loss ")

            graph, num_features, ad_concat = load_ST_dataset_hard(dataset_name=dataset_name, pi=global_PI, section_ids=section_ids, hvgs=args.hvgs, st_data_dir=st_data_dir)
            args.num_features = num_features
            x = graph.ndata["feat"]

            # args.mask_rate = args_mask_rate
            # args.remask_rate = args_remask_rate
            print(args)
            model = build_model_ST(args)
            print(model)
            model.to(device)
            optimizer = create_optimizer(optim_type, model, lr, weight_decay)

            if use_scheduler:
                logging.info("Use scheduler")
                scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
            else:
                scheduler = None

            model.to(device)
            graph = graph.to(device)
            x = x.to(device)
            # print(ad_concat[0])
            model, ad_concat_1 = MG(model, graph, x, optimizer, max_epoch, device, ad_concat, scheduler, logger=logger, key_="MG")
            # print(ad_concat_1)
            # print(ad_concat_1.obsm["MG"])
            ari_ = visualization_umap_spatial(ad_temp=ad_concat_1, section_ids=section_ids, exp_fig_dir=exp_fig_dir, dataset_name=dataset_name, num_iter=counter, identifier="stage1", num_class=args.num_class, use_key="MG")
            ari_1_pre.append(ari_[0])
            ari_2_pre.append(ari_[1])
            if logger is not None:
                logger.log({"slice1_ari_pre": ari_[0], "slice2_ari_pre": ari_[1]})
            # print(section_id)
            print(section_ids[0], ', ARI = %01.3f' % ari_[0])
            print(section_ids[1], ', ARI = %01.3f' % ari_[1])
            # exit(-1)
            """train with MSSL + triplet loss"""
            model, ad_concat_2 = MG_triplet(model, graph, x, optimizer, max_epoch_triplet, device, adata_concat_=ad_concat_1, pis_list_=[global_PI], scheduler=scheduler, logger=logger, key_="MG_triplet")
            ari_ = visualization_umap_spatial(ad_temp=ad_concat_2, section_ids=section_ids, exp_fig_dir=exp_fig_dir, dataset_name=dataset_name, num_iter=counter, identifier="stage2", num_class=args.num_class, use_key="MG_triplet")
            counter += 1
            ari_1.append(ari_[0])
            ari_2.append(ari_[1])
            if logger is not None:
                logger.log({"slice1_ari_after": ari_[0], "slice2_ari_after": ari_[1]})
            print(section_ids[0], ', ARI = %01.3f' % ari_[0])
            print(section_ids[1], ', ARI = %01.3f' % ari_[1])
        if logger is not None:
            logger.finish()

    return None


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args_ST()
    a_ari = main(args)
