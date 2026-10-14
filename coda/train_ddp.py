# 임시 파일 
""" 0116
# Singlegpu: python (to run multiple cond in a single gpu without socket issue)
# no gradaccumulation, correct order of train_transform
# add argparser (yaml is additional config for experimental settings)
# 

# CUDA_VISIBLE_DEVICE --> shell script
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_ids) # set specific GPU
"""

import os
import sys
import argparse
import yaml
from datetime import datetime, timedelta

# Pytorch
import torch
import torch.nn as nn


# Model
from modules.get_model import get_model
from modules.get_lrsch import get_scheduler, get_optimizer
from modules.get_data import get_trainval_dataset, get_train_loader, get_val_loader
from modules.valid import valid_ddp
from modules.train import train_ddp
from modules.frequency import fft3channels_batch, set_mask
from modules.utils import *

# DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# wandb
import wandb

# argparser
parser = argparse.ArgumentParser(description='FC-RAQ_train')
parser = argparse.ArgumentParser(description="Arguments for DDP Training")
parser.add_argument('--yaml', type=str, default='config.yaml', help='Path to yaml file for additional configurations')

### Setup
def parse_yaml(args):
    with open(args.yaml) as f: 
        cfg = yaml.load(f, Loader=yaml.FullLoader) # dictionary
        # yaml (dict) --> argparse (namespace)
    try:
        {setattr(args, key, cfg[key]) for key in cfg.keys()}
    except Exception as e:
        sys.exit(1)
        
    return args

def setup(args): 
    # Date
    now = datetime.utcnow()
    time_gap = timedelta(hours=9)
    now += time_gap
    args.now = now.strftime("%Y%m%d_%H%M")  # Get the current date


    
    # DDP
    # 환경 변수 확인
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        assert int(args.world_size)==int(torch.cuda.device_count())
    else:
        args.world_size = torch.cuda.device_count()

    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0

    # DDP 초기화
    dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.world_size)

    # 배치 크기 조정
    total_batch_size = args.batch_size * args.world_size * args.update_freq
    
    # Save path
    lr = args.kwargs_optim['lr']
    args.setting = f'{args.optim}_lr{lr}_{args.lrsch}_freq{args.freq}'
    args.save_path = os.path.join(args.save_path, args.setting)
    if (not os.path.exists(args.save_path)) and (args.local_rank == 0):
        os.makedirs(args.save_path)
        print(f"[INFO] Makedirs {args.save_path}")
    else:
        pass
    
    print(f"[INFO] Rank {args.local_rank}/{args.world_size}: \n\t model {args.model_arch}, \n\t bits: {args.q_bits_a}, \n\t ckpt_path: {args.save_path}")
    
    if args.q_scheme == 'lq': args.find_unused_parameters = True
    else:  args.find_unused_parameters = False
    
    return args

def initialize_wandb(args):
    wandb.init(
        project=args.project_name,  # Set your project name
        name=args.setting,          # Use the experiment setting as the run name
        config=vars(args)           # Log all arguments as config
    )
    print("[INFO] Initialized wandb")

def main():
    # Parser
    print("[STATUS] Parse arguments\n")
    args = parser.parse_args() # load argparser
    args = parse_yaml(args) # load yaml as args 
    args = setup(args)
    
    if args.use_wandb and args.local_rank == 0:
        print("[INFO] Initialize wandb on rank 0")
        initialize_wandb(args)
    
    ### Dataset & DataLoader
    print("[STATUS] Start setting dataset")
    train_set, val_set, num_classes, source_dataset = get_trainval_dataset(args)
    train_sampler = DistributedSampler(train_set, shuffle=True)
    val_sampler = DistributedSampler(val_set, shuffle=False)
    train_loader = get_train_loader(args, train_set, train_sampler=train_sampler)
    val_loader = get_val_loader(args, val_set, val_sampler=val_sampler)
    train_loader_len = len(train_loader)
    print(f"[INFO] Rank {args.local_rank}/{args.world_size}: Local batch size {args.batch_size}, update frequency: {args.update_freq}, train_sampler {train_sampler.num_samples}, total {train_sampler.total_size}, train_loader {train_loader_len}")
    
    ### Model
    print(f"[STATUS] Start setting model: {args.model_arch}")
    with torch.device(f"cuda:{args.local_rank}"):
        model = get_model(args, num_classes)
        ### Checkpoint 
        if args.pretrained == True and args.checkpoint_path is not None:
            assert os.path.exists(args.checkpoint_path), f"[ERROR] No checkpoint in {args.checkpoint_path}"
            ckpt = torch.load(args.checkpoint_path)
            if 'state_dict' in ckpt.keys(): model.load_state_dict(ckpt['state_dict'])
            else: model.load_state_dict(ckpt['model'])
            print(f"[STATUS] Load checkpoint from {args.checkpoint_path}")
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    model.cuda(args.local_rank)

    
    ### DDP
    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters= args.find_unused_parameters)

    ### Optimizer & Scheduler & Criterion
    optimizer = get_optimizer(args.optim, model.parameters(), args.kwargs_optim)
    if args.lrsch is not None:
        lr_scheduler = get_scheduler(args.lrsch, optimizer, args.kwargs_lrsch, train_loader_len)

    criterion = nn.CrossEntropyLoss().cuda()        

    if args.pretrained == True and args.checkpoint_path is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.last_epoch = (args.last_epoch - 1) * train_loader_len
        print(f"[STATUS] Load optim & scheduler from {args.checkpoint_path}")


    ### Train & Validate
    print("[STATUS] Start training")
    for epoch in range(args.last_epoch + 1, args.epochs + 1):
        if args.freq > 0 and args.use_fft:
            args.mask = set_mask(args.input_size, 0, args.freq).to(args.device) # low: 0~freq, high: freq~input_size

        # train for one epoch
        train_sampler.set_epoch(epoch)
        train_loss, train_acc = train_ddp(train_loader, model, criterion, optimizer, epoch, args)
    
        # step lrsch
        if args.lrsch is not None:
            lr_scheduler.step()


        # validate
        dist.barrier() # wait until save all params in main procss

        if epoch % args.val_freq == 0:
            # save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, 
                f"{args.save_path}/e{epoch}.pt")
            
            val_model = get_model(args, num_classes)
            ckpt = torch.load(f"{args.save_path}/e{epoch}.pt")
            val_model.load_state_dict(ckpt['state_dict'], strict=True)
            val_model.cuda(args.local_rank)
            val_model = DDP(val_model, device_ids=[args.local_rank], find_unused_parameters=args.find_unused_parameters)
            val_sampler.set_epoch(epoch)
            
            val_loss, val_acc = valid_ddp(val_loader, val_model, criterion, epoch, args)
            
            if args.use_wandb and args.local_rank == 0:
                wandb.log({"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc})
        
if __name__ == '__main__':
    main()