"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from torch.utils.data import DataLoader, DistributedSampler
from data_utils import Dataset_, Dataset_0, Dataset_api
from evaluation import calculate_EER, calculate_EER_evel, calculate_EER_DCF_eval
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
import logging
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import warnings
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader
from typing import List, Tuple
from tqdm import tqdm

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
import os
os.environ["NUMBA_DISABLE_CACHE"] = "1"

def main(args: argparse.Namespace) -> None:

    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """


    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        logging.info("=================start=======================")
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]

    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    database_path = os.path.join(config["data_root_path"])

    # define model related paths
    model_tag = "{}_ep{}_bs{}".format(
        config["model_config"]['architecture'],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    if args.rank == 0:
        writer = SummaryWriter(model_tag)
    else:
        writer = None
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)
    for name, param in model.named_parameters():
        if param.is_sparse:
            print(f"Sparse tensor found: {name}")

    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        args.world_size,  args.rank, database_path,args.seed, config,args)

    # evaluates pretrained model and exit script
    if args.eval:

        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        logging.info("Model loaded : {}".format(config["model_path"]))
        logging.info("Start evaluation...")
        eval_f1 = evaluation_file_ddp('eval',eval_loader, model, device)


    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)
    if args.rank == 0:
        best_dev_f1 = 0.
        best_eval_f1 = 0.
        n_swa_update = 0  # number of snapshots of model to use in SWA
    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)
    start_epoch = 0
    n_swa_update = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') is not None and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        n_swa_update = checkpoint.get('swa_n', 0)
        logging.info(f"Resumed training from epoch {start_epoch}")


    # Training
    for epoch in range(start_epoch, config["num_epochs"]):
        if args.rank == 0:
            logging.info("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        # 同步所有GPU的 loss
        loss_tensor = torch.tensor(running_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= dist.get_world_size()
        running_loss = loss_tensor.item()

        if args.rank == 0:
            writer.add_scalar("training loss", running_loss, epoch)
            logging.info(f"training loss: {running_loss:.5f}, epoch: {epoch}")
        if args.rank == 0:
            dev_f1 = evaluation_file_ddp('dev', dev_loader, model, device)
            logging.info("\ntraining loss:{:.5f}, dev_eer: {:.3f}".format(
                running_loss, dev_f1))
            writer.add_scalar("dev_eer", dev_f1, epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'swa_n': n_swa_update,
            }, model_save_path / "latest_checkpoint.pth")

            if best_dev_f1 >= dev_f1:
                logging.info(f"best model found at epoch {epoch}")
                best_dev_f1 = dev_f1


                torch.save(model.state_dict(),
                           model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_f1))
                logging.info("Saving epoch {} for swa".format(epoch))
                optimizer_swa.update_swa()
                n_swa_update += 1
                writer.add_scalar("best_dev_eer", best_dev_f1, epoch)
                logging.info(f"best_dev_eer: {best_dev_f1:.4f}%, at epoch {epoch}")


        if best_dev_f1 >= dev_f1 and str_to_bool(config["eval_all_best"]):

            eval_f1 = evaluation_file_ddp('eval', eval_loader, model, device)
            log_text = "epoch{:03d}, ".format(epoch)
            if eval_f1 < best_eval_f1:
                log_text += "best eval eer, {:.4f}%".format(eval_f1)
                best_eval_f1 = eval_f1
                torch.save(model.state_dict(),
                           model_save_path / "best.pth")
            if len(log_text) > 0:
                logging.info(log_text)

    logging.info("==================Start final evaluation====================")
    epoch += 1

    if args.rank == 0 and n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)

    f1 = evaluation_file_ddp('eval',eval_loader, model, device)
    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    if f1 <= best_eval_f1:
        best_eval_f1 = f1
        torch.save(model.state_dict(),
                   model_save_path / "best.pth")
    logging.info("Exp FIN. EER: {:.3f}".format(best_eval_f1))


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config, device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logging.info("no. model params:{}".format(nb_params))

    return model


def get_loader(
world_size,rank,
        database_path: str,
        seed: int,
        config: dict,
        args) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""



    trn_list_path = os.path.join(config["data_root_path"], config["label_train"])
    dev_trial_path = os.path.join(config["data_root_path"], config["label_dev"])
    eval_trial_path = os.path.join(config["data_root_path"], config["label_eval"])

    whisper = False
    if "whisper" in config["model_config"]['architecture']:
        whisper = True

    train_set = Dataset_api(dir_meta=trn_list_path, root_dir=config["data_root_path"],args=args,whisper = whisper)
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)

    trn_loader = DataLoader(train_set,
                              batch_size=config["batch_size"],
                              sampler=train_sampler,
                            # num_workers=4,
                            worker_init_fn=seed_worker,
                              drop_last=True,
                              pin_memory=True,
                              generator=gen)




    dev_set = Dataset_api(dir_meta=dev_trial_path, root_dir=config["data_root_path"],args=args,whisper = whisper)
    dev_sampler = DistributedSampler(dev_set, num_replicas=world_size, rank=rank, shuffle=False)

    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            sampler=dev_sampler,
                            # num_workers=4,
                            worker_init_fn=seed_worker,
                            # shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    eval_set = Dataset_api(dir_meta=eval_trial_path, root_dir=config["data_root_path"],args=args,whisper = whisper)
    eval_sampler = DistributedSampler(eval_set, num_replicas=world_size, rank=rank, shuffle=False)

    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             sampler=eval_sampler,
                             # num_workers=4,
                             worker_init_fn=seed_worker,
                             drop_last=False,
                             # shuffle = False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader




def evaluation_file_ddp(
    type,
    data_loader: DataLoader,
    model,
    device: torch.device,
    threshold: float = 0.5
):
    """Perform evaluation and return metrics with unseen class."""
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_x, batch_y, file_names in tqdm(data_loader, total=len(data_loader),
                                                 desc=type, dynamic_ncols=True, ascii=True):
            batch_x = batch_x.to(device)

            # 模型输出 (B,21)，假设已经是 softmax 概率
            _, batch_out = model(batch_x)   # (B,21)
            probs = torch.softmax(batch_out, dim=-1)  # 确保是概率

            max_probs, preds = probs.max(dim=-1)  # (B,)
            preds = preds.cpu().numpy()
            max_probs = max_probs.cpu().numpy()

            batch_y = batch_y.cpu().numpy()

            # 扩展为 22 类
            preds_ext = []
            for i in range(len(preds)):
                if max_probs[i] < threshold:
                    preds_ext.append(21)  # unseen 类索引
                else:
                    preds_ext.append(preds[i])
            all_preds.extend(preds_ext)
            all_labels.extend(batch_y)

    # === 计算总体指标 ===
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # 每类指标
    report = classification_report(all_labels, all_preds, labels=list(range(22)),
                                   target_names=[f"class_{i}" for i in range(22)],
                                   zero_division=0, digits=4)

    logging.info(f"== {type} Metrics ==")
    logging.info(f"Precision (macro): {precision:.4f}")
    logging.info(f"Recall    (macro): {recall:.4f}")
    logging.info(f"F1        (macro): {f1:.4f}")
    logging.info("\n== Per-class Report ==")
    logging.info(report)

    return precision, recall, f1, report

def evaluation_file_ddp_save(
    type,
    data_loader: DataLoader,
    model,
    device: torch.device,
    threshold: float = 0.5,
    feature_save_path: str = './feature/api_tracing'
):
    """Perform evaluation and return metrics with unseen class.
       Also save features grouped by class if feature_save_path is given.
    """
    model.eval()

    all_preds, all_labels = [], []
    all_features_by_class = {i: [] for i in range(22)}  # 保存每类特征

    with torch.no_grad():
        for batch_x, batch_y, file_names in tqdm(data_loader, total=len(data_loader),
                                                 desc=type, dynamic_ncols=True, ascii=True):
            batch_x = batch_x.to(device)

            # 模型输出
            feats, logits = model(batch_x)   # feats: (B, D), logits: (B,21)
            probs = torch.softmax(logits, dim=-1)  # (B,21)

            max_probs, preds = probs.max(dim=-1)  # (B,)
            preds = preds.cpu().numpy()
            max_probs = max_probs.cpu().numpy()

            batch_y = batch_y.cpu().numpy()
            feats = feats.cpu().numpy()

            # 扩展为 22 类
            preds_ext = []
            for i in range(len(preds)):
                if max_probs[i] < threshold:
                    preds_ext.append(21)  # unseen 类索引
                else:
                    preds_ext.append(preds[i])
            all_preds.extend(preds_ext)
            all_labels.extend(batch_y)

            # 保存特征
            for i, pred_class in enumerate(preds_ext):
                all_features_by_class[pred_class].append(feats[i])

    # === 计算总体指标 ===
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # 每类指标
    report = classification_report(all_labels, all_preds, labels=list(range(22)),
                                   target_names=[f"class_{i}" for i in range(22)],
                                   zero_division=0, digits=4)

    logging.info(f"== {type} Metrics ==")
    logging.info(f"Precision (macro): {precision:.4f}")
    logging.info(f"Recall    (macro): {recall:.4f}")
    logging.info(f"F1        (macro): {f1:.4f}")
    logging.info("\n== Per-class Report ==")
    logging.info(report)

    # === 保存特征到文件 ===
    if feature_save_path is not None:
        os.makedirs(feature_save_path, exist_ok=True)
        for cls_id, feat_list in all_features_by_class.items():
            if len(feat_list) == 0:
                continue
            feat_array = np.stack(feat_list, axis=0)  # (N, D)
            np.save(os.path.join(feature_save_path, f"class_{cls_id}.npy"), feat_array)
        logging.info(f"Features saved to {feature_save_path}")

    return precision, recall, f1, report


def ddp_gather_object(obj, rank, world_size):
    """Gather arbitrary picklable objects from all ranks to rank 0"""
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(obj, gathered, dst=0)
    return gathered  # Only valid on rank 0






def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    # weight = torch.FloatTensor([0.1, 0.9]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = nn.CrossEntropyLoss()


    # for batch_x, batch_y,_ in trn_loader:
    for batch_x, batch_y,_ in tqdm(trn_loader, desc="Training", dynamic_ncols=True, ascii=True):
        batch_size = batch_x.size(0)
        num_total += batch_size

        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))
    #######################################################################################
        # if num_total > 800:
        #     break
    #######################################################################################
    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    print('=====================')
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser(description="detection system")
    parser.add_argument("--outfile",
                        dest="outfile",
                        type=str,
                        help="configuration file",
                        required=True)


    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )

    parser.add_argument("--world-size", default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument("--rank", default=-1, type=int,
                        help='ranking within the nodes')
    parser.add_argument("--dist-url", default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')



    parser.add_argument("--local-rank", default=-1, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")

    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument("--resume",
                        type=str,
                        default=None,
                        help="path to checkpoint to resume training")
    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=0,
                        help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                              5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .default=0]')

    # LnL_convolutive_noise parameters
    parser.add_argument('--nBands', type=int, default=5,
                        help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20,
                        help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000,
                        help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100,
                        help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000,
                        help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10,
                        help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100,
                        help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0,
                        help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0,
                        help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5,
                        help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20,
                        help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5,
                        help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10,
                        help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2,
                        help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10,
                        help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40,
                        help='Maximum SNR value for coloured additive noise.[defaul=40]')

    ##===================================================Rawboost data augmentation ======================================================================#


    logging.basicConfig(
        filename=parser.parse_args().outfile,
        level=logging.INFO,
        format='%(asctime)s %(message)s',
    )

    main(parser.parse_args())
