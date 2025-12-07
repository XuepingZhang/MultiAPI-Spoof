"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import datetime
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
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from torch.utils.data import DataLoader, DistributedSampler
from data_utils import Dataset_, Dataset_0
from evaluation import calculate_EER, calculate_EER_evel, calculate_EER_DCF_eval
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
import logging
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import warnings
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

import os
os.environ["NUMBA_DISABLE_CACHE"] = "1"

def main(args: argparse.Namespace) -> None:

    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(seconds=600))
    # ----- 固定随机种子 -----
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 确保 cudnn 可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 同步所有进程，确保种子一致
    dist.barrier()

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


    if args.eval:

        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        logging.info("Model loaded : {}".format(config["model_path"]))
        logging.info("Start evaluation...")
        local_results, spoof_features_np = produce_evaluation_file_ddp_save(
            'eval', eval_loader, model, device
        )


        # 保存本 rank 的分数结果
        os.makedirs("./temp", exist_ok=True)
        save_path_rank = f"./temp/eval_score_rank{args.rank}.txt"
        with open(save_path_rank, "w") as f:
            for file_name, score, label in local_results:
                f.write(f"{file_name} {score} {label}\n")


        dist.barrier()
        # ---------- 合并 txt & 计算指标 ----------
        if args.rank == 0:
            with open(eval_score_path, "w") as fout:
                for i in range(args.world_size):
                    temp_path = f"./temp/eval_score_rank{i}.txt"
                    with open(temp_path) as fin:
                        fout.write(fin.read())
            logging.info("Scores saved to {}".format(eval_score_path))

            unseen_types = {"soundful_10s", "tor", "udio_10s", "voc", "wild-ai", "beatoven_10s", "xttsv2", "yaya", "zijie","C7"}

            metric_path = model_tag / "metrics"
            os.makedirs(metric_path, exist_ok=True)
            calculate_EER_DCF_eval(
                cm_scores_file=eval_score_path,
                output_file=metric_path / "results_eval_only.txt",
                unseen_types=unseen_types)

            sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)
    if args.rank == 0:
        best_dev_eer = 100.
        best_eval_eer = 100.
        no_impro = 0
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
        local_results = produce_evaluation_file_ddp('dev',dev_loader, model, device)
        os.makedirs("./temp", exist_ok=True)
        save_path_rank = f"./temp/dev_score_rank{args.rank}.txt"
        with open(save_path_rank, "w") as f:
            for file_name, score, label in local_results:
                f.write(f"{file_name} {score} {label}\n")
        dist.barrier()
        if args.rank == 0:
            with open(metric_path/"dev_score.txt", "w") as fout:
                for i in range(args.world_size):
                    temp_path = f"./temp/dev_score_rank{i}.txt"
                    with open(temp_path) as fin:
                        fout.write(fin.read())
            logging.info("Scores saved to {}".format(metric_path/"dev_score.txt"))

            unseen_types = {"soundful_10s", "tor", "udio_10s", "voc", "wild-ai", "beatoven_10s", "xttsv2", "yaya", "zijie","C7"}


            dev_eer, overall_min_dcf, overall_act_dcf = calculate_EER_DCF_eval(
                cm_scores_file=metric_path/"dev_score.txt",
                output_file=metric_path / "results_dev_ep{}_th.txt".format(epoch),
                unseen_types=unseen_types
            )
            logging.info("\ntraining loss:{:.5f}, dev_eer: {:.3f}".format(
                running_loss, dev_eer))
            writer.add_scalar("dev_eer", dev_eer, epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'swa_n': n_swa_update,
            }, model_save_path / "latest_checkpoint.pth")

            if best_dev_eer >= dev_eer:
                logging.info(f"best model found at epoch {epoch}")
                best_dev_eer = dev_eer
                no_impro = 0

                torch.save(model.state_dict(),
                           model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))
                logging.info("Saving epoch {} for swa".format(epoch))
                optimizer_swa.update_swa()
                n_swa_update += 1
                writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
                logging.info(f"best_dev_eer: {best_dev_eer:.4f}%, at epoch {epoch}")
            else:
                no_impro += 1
                if no_impro > 10:
                    break

        #eval
        best_dev_eer_list = [best_dev_eer if args.rank == 0 else None]
        dist.broadcast_object_list(best_dev_eer_list, src=0)
        best_dev_eer_all_rank = best_dev_eer_list[0]

        dev_eer_list = [dev_eer if args.rank == 0 else None]
        dist.broadcast_object_list(dev_eer_list, src=0)
        dev_eer_all_rank = dev_eer_list[0]

        if best_dev_eer_all_rank >= dev_eer_all_rank and str_to_bool(config["eval_all_best"]):

            local_results = produce_evaluation_file_ddp('eval', eval_loader, model, device)
            os.makedirs("./temp", exist_ok=True)
            save_path_rank = f"./temp/eval_score_rank{args.rank}.txt"
            with open(save_path_rank, "w") as f:
                for file_name, score, label in local_results:
                    f.write(f"{file_name} {score} {label}\n")
            dist.barrier()
            if args.rank == 0:
                with open(eval_score_path, "w") as fout:
                    for i in range(args.world_size):
                        temp_path = f"./temp/eval_score_rank{i}.txt"
                        with open(temp_path) as fin:
                            fout.write(fin.read())
                logging.info("Scores saved to {}".format(eval_score_path))

                unseen_types = {"soundful_10s", "tor", "udio_10s", "voc", "wild-ai", "beatoven_10s", "xttsv2", "yaya", "zijie","C7"}
                metric_path = model_tag / "metrics"
                eval_eer, overall_min_dcf, overall_act_dcf = calculate_EER_DCF_eval(
                    cm_scores_file=eval_score_path,
                    output_file=metric_path / "eval_results_epo{}_th_eval.txt".format(epoch),
                    unseen_types=unseen_types)

                log_text = "epoch{:03d}, ".format(epoch)
                if eval_eer < best_eval_eer:
                    log_text += "best eval eer, {:.4f}%".format(eval_eer)
                    best_eval_eer = eval_eer

                    torch.save(model.state_dict(),
                               model_save_path / "best.pth")
                if len(log_text) > 0:
                    logging.info(log_text)



    logging.info("==================Start final evaluation====================")
    epoch += 1

    if args.rank == 0 and n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    local_results = produce_evaluation_file_ddp('eval',eval_loader, model, device)
    os.makedirs("./temp", exist_ok=True)
    save_path_rank = f"./temp/eval_score_rank{args.rank}.txt"
    with open(save_path_rank, "w") as f:
        for file_name, score, label in local_results:
            f.write(f"{file_name} {score} {label}\n")
    dist.barrier()
    if args.rank == 0:
        with open(eval_score_path, "w") as fout:
            for i in range(args.world_size):
                temp_path = f"./temp/eval_score_rank{i}.txt"
                with open(temp_path) as fin:
                    fout.write(fin.read())
        logging.info("Scores saved to {}".format(eval_score_path))

        unseen_types = {"soundful_10s", "tor", "udio_10s", "voc", "wild-ai", "beatoven_10s", "xttsv2", "yaya", "zijie","C7"}

        eval_eer, overall_min_dcf, overall_act_dcf= calculate_EER_DCF_eval(
            cm_scores_file=eval_score_path,
            output_file=metric_path/"results_eval_final.txt",
            unseen_types=unseen_types
        )


        torch.save(model.state_dict(),
                   model_save_path / "swa.pth")

        if eval_eer <= best_eval_eer:
            best_eval_eer = eval_eer
            torch.save(model.state_dict(),
                       model_save_path / "best.pth")
        logging.info("Exp FIN. EER: {:.3f}".format(best_eval_eer))


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

    train_set = Dataset_0(dir_meta=trn_list_path, root_dir=config["data_root_path"],args=args,whisper = whisper)
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




    dev_set = Dataset_0(dir_meta=dev_trial_path, root_dir=config["data_root_path"],args=args,whisper = whisper)
    dev_sampler = DistributedSampler(dev_set, num_replicas=world_size, rank=rank, shuffle=False)

    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            sampler=dev_sampler,
                            # num_workers=4,
                            worker_init_fn=seed_worker,
                            # shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    eval_set = Dataset_0(dir_meta=eval_trial_path, root_dir=config["data_root_path"],args=args,whisper = whisper)
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

def produce_evaluation_file_ddp(
    type,
    data_loader: DataLoader,
    model,
    device: torch.device) -> List[Tuple[str, float, str]]:
    """Perform evaluation and return results for aggregation."""
    model.eval()
    label_map = {1: 'bonafide', 0: 'spoof'}

    results = []

    for batch_x, batch_y, file_names in tqdm(data_loader, total=len(data_loader), desc=type, dynamic_ncols=True, ascii=True):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            h, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()

        for file_name, score, y in zip(file_names, batch_score, batch_y):
            results.append((file_name, float(score), label_map[int(y)]))

        # 可选限制，调试用
        # if len(results) > 100:
        #     break

    return results

from typing import List, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

def produce_evaluation_file_ddp_save(
    type: str,
    data_loader: DataLoader,
    model,
    device: torch.device
) -> Tuple[List[Tuple[str, float, str]], np.ndarray]:
    """Perform evaluation and return results + spoof features for aggregation."""
    model.eval()
    label_map = {1: 'bonafide', 0: 'spoof'}

    results = []
    spoof_features = []   # 存储所有 label=0 的特征

    for batch_x, batch_y, file_names in tqdm(data_loader, total=len(data_loader),
                                             desc=type, dynamic_ncols=True, ascii=True):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            h, batch_out = model(batch_x)  # h shape: [B, D]
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()

        # 遍历 batch
        for i, (file_name, score, y) in enumerate(zip(file_names, batch_score, batch_y)):
            results.append((file_name, float(score), label_map[int(y)]))
            if int(y) == 0:  # spoof
                spoof_features.append(h[i].detach().cpu().numpy())

        # 可选调试
        # if len(results) > 100:
        #     break

    # 转为 numpy (N_spoof, D)，如果没有 spoof 样本就返回空数组
    if len(spoof_features) > 0:
        spoof_features_np = np.stack(spoof_features, axis=0)
    else:
        spoof_features_np = np.empty((0, h.shape[1]))

    return results, spoof_features_np

def ddp_gather_object(obj, rank, world_size):
    """Gather arbitrary picklable objects from all ranks to rank 0"""
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(obj, gathered, dst=0)
    return gathered  # Only valid on rank 0

def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()

    num_total = 0
    label_map = {1:'bonafide', 0:'spoof'}
    with open(save_path, "w") as fh:
        for batch_x, batch_y,file_names in tqdm(data_loader, total=len(data_loader), desc="eval", dynamic_ncols=True, ascii=True):
            num_total += 1
            batch_x = batch_x.to(device)
            num_total += 1
            with torch.no_grad():
                _, batch_out = model(batch_x)
                batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            # add outputs
            for file_name, score, y in zip(file_names, batch_score, batch_y):
                fh.write("{} {} {}\n".format(file_name, score, label_map[int(y)]))
            # if num_total>100:
            #     break
    logging.info("Scores saved to {}".format(save_path))





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
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    # weight = torch.FloatTensor([0.2, 0.8]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)


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
    parser.add_argument("--node-rank", default=-1, type=int,
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
