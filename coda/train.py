import os
import sys
import argparse
import yaml
from datetime import datetime, timedelta

# wandb
import wandb

# Pytorch
import torch
import torch.nn as nn

# Model
from modules.get_model import get_model
from modules.get_lrsch import get_scheduler, get_optimizer
from modules.get_data import get_trainval_dataset, get_train_loader, get_val_loader
from modules.valid import valid
from modules.train import train
from modules.utils import *
from modules.frequency import set_mask
# argparser
parser = argparse.ArgumentParser(description='FC-RAQ_train')
parser = argparse.ArgumentParser(description="Arguments for CIFAR10 EfficientNet-B0 Training")
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

    # Check GPU
    if torch.cuda.is_available():
        args.device = 'cuda'
        args.num_gpus = torch.cuda.device_count()
        args.gpu_ids = torch.cuda.current_device()
    else:
        args.device = 'cpu'
        args.num_gpus = 0
        args.gpu_ids = None

    # Save path
    args.total_batch_size = int(args.batch_size) * int(args.num_gpus) * int(args.update_freq)
    lr = args.kwargs_optim['lr']
    args.setting = f'{args.optim}_lr{lr}_{args.lrsch}_b{args.total_batch_size}_resize{args.input_size}_freq{args.freq}'
    args.save_path = os.path.join(args.save_path, args.setting)
    if (not os.path.exists(args.save_path)):
        os.makedirs(args.save_path)
        print(f"[INFO] Makedirs {args.save_path}")
    else:
        pass
    

    
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
    
    ### wandb
    if args.use_wandb:
        print("[INFO] Initialize wandb on GPU ", args.gpu_ids)
        initialize_wandb(args)
        print(f"[INFO] Arguments \n\t project_name: {args.project_name}, \n\t Wandb run_name: {wandb.run.name}, \n\t save_path: {args.save_path}, \n\t device: {args.device}, num_gpus: {args.num_gpus}, gpu_ids (torch): {args.gpu_ids} \n\t model {args.model_arch}, \n\t bits: {args.q_bits_a}, \n\t resize: {args.input_size}, \n\t use fft: {args.freq>0}\n")
    else:
        print(f"[INFO] Arguments \n\t project_name: {args.project_name}, \n\t save_path: {args.save_path}, \n\t device: {args.device}, num_gpus: {args.num_gpus}, gpu_ids (torch): {args.gpu_ids} \n\t model {args.model_arch}, \n\t bits: {args.q_bits_a}, \n\t resize: {args.input_size}, \n\t use fft: {args.freq>0}\n")

    ### Dataset & DataLoader
    print("[STATUS] Start setting dataset")
    train_set, val_set, num_classes, source_dataset = get_trainval_dataset(args)
    train_loader = get_train_loader(args, train_set, train_sampler=None)
    val_loader = get_val_loader(args, val_set, val_sampler=None)
    print(f"[INFO] batch size {args.batch_size} total {len(train_set)}, train_loader {len(train_loader)}")
    
    ### Model
    print(f"[STATUS] Start setting model: {args.model_arch}")
    model = get_model(args, num_classes)
    model.to(args.device)
    
    ### Checkpoint 
    if args.pretrained == True:
        ckpt = torch.load(args.checkpoint_path)
        if 'state_dict' in ckpt.keys(): model.load_state_dict(ckpt['state_dict'])
        else: model.load_state_dict(ckpt['model'])
        print(f"[STATUS] Load checkpoint from {args.checkpoint_path}")

    ### Optimizer & Scheduler & Criterion
    optimizer = get_optimizer(args.optim, model.parameters(), args.kwargs_optim)
    if args.lrsch is not None:
        lr_scheduler = get_scheduler(args.lrsch, optimizer, args.kwargs_lrsch, len(train_loader))

    criterion = nn.CrossEntropyLoss().to(args.device)    

    ### Resume training
    if args.pretrained == True:
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.last_epoch = (args.last_epoch - 1) * len(train_loader)
        print(f"[STATUS] Load optim & scheduler from {args.checkpoint_path}")


    ### Train & Validate
    print("[STATUS] Start training")
    for epoch in range(args.last_epoch + 1, args.epochs + 1):
        # train for one epoch
        if args.freq > 0 and args.use_fft:
            args.mask = set_mask(args.input_size, 0, args.freq).to(args.device) # low: 0~freq, high: freq~input_size

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args)
        if args.use_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                            "lr": optimizer.param_groups[0]['lr']})
    
        # step lrsch
        if args.lrsch is not None:
            if args.lrsch == 'exp':
                if (epoch+1) % 2.4 < 1 :
                    lr_scheduler.step()
                else:
                    pass
            else:
                lr_scheduler.step()

        # validate
        if (epoch == 1) or (epoch + 1) % args.val_freq == 0:  
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, 
                f"{args.save_path}/e{epoch}.pt")          
            val_model = get_model(args, num_classes)
            ckpt = torch.load(f"{args.save_path}/e{epoch}.pt")
            val_model.load_state_dict(ckpt['state_dict'], strict=True)
            val_model.to(args.device)
            
            val_loss, val_acc = valid(val_loader, val_model, criterion, epoch, args)
            if args.use_wandb:
                wandb.log({"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc})
        
if __name__ == '__main__':
    main()