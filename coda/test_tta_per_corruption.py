"""
With save_results
"""
# Import Library
import sys
import yaml
import os
import argparse
from copy import deepcopy
import math

import numpy as np
import torch
import torch.nn as nn

# Model
from modules.get_model import get_model
from modules.get_data import get_test_dataset, get_test_loader
from modules.valid import valid
from modules.utils import *

# tta
from tta_algorithms import coda, tent, norm, sar, tent_coda, sar_coda
from tta_algorithms.sam import SAM

# setup in test_tta.py
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
    args.device = torch.device(f'cuda' if torch.cuda.is_available() else torch.device('cpu'))    
    
    if args.project_name:
        args.project_name = args.model_arch.split('_')[-1] + '_' + args.q_scheme  + '_' + args.project_name
    else:
        args.project_name = args.model_arch.split('_')[-1] + '_' + args.q_scheme 
    args.train_freq = args.ckpt_path.split('/')[-2]
   
    if args.q_bits_a:      # imagenet_resnet18_lq_2bit_seed17/r0r0/
        args.project_path = f'{args.dataset}_{args.project_name}_{args.q_bits_a}bit_seed{args.random_seed}/{args.train_freq}/{args.tta_method}' # imagenet_resnet18_fp_seed17/r0r0/norm_v2
    else:     # imagenet_resnet18_fp_seed17/r0r0/...
        args.project_path = f'{args.dataset}_{args.project_name}_seed{args.random_seed}/{args.train_freq}/{args.tta_method}' # imagenet_resnet18_fp_seed17/r0r0/norm_v2

    args.save_path = f'./checkpoint/{args.project_path}/' # gaussian.pth
    args.log_path = f'./logs/{args.project_path}' # val_result.csv, adapt_result.csv

    if (not os.path.exists(args.save_path)):
        os.makedirs(args.save_path)     
    else:
        pass
    
    if (not os.path.exists(args.log_path)):
        os.makedirs(args.log_path)
    else:
        pass    
    
    print(f"[INFO] PROJECT_PATH : {args.project_path} \n\t LOG PATH : {args.log_path} \n\t TTA_METHOD : {args.tta_method} \n\t CORRUPTION {args.corr_type} \n\t Q_SCHEME : {args.q_scheme} \n\t Q_BITS_W : {args.q_bits_w} \n\t Q_BITS_A : {args.q_bits_a}")   
    
    return args

### Main
def main(args):   
    ### Random seed
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
    
    ### Dataset
    print("[STATUS] Start setting dataset")
    test_set, num_classes, source_dataset, index_filter = get_test_dataset(args)
    test_loader = get_test_loader(args, test_set)
    
    ### Set model
    print(f"[STATUS] Start setting model: {args.model_arch}")
    
    if args.q_scheme == 'fp': args.q_scheme = None
    if args.q_bits_a < 32: args.q_bits_a = int(args.q_bits_a) 
    else: args.q_bits_a = None
    if args.q_bits_w < 32: args.q_bits_w = int(args.q_bits_w) 
    else: args.q_bits_w = None
    
    tta_model = get_model(args, num_classes)
    tta_model.to(args.device)
        
    ### Checkpoint
    print(f"[INFO] Load checkpoint from {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path)
    if 'state_dict' in ckpt.keys(): 
        if 'module' in list(ckpt['state_dict'].keys())[0]:  # DDP model
            new_state_dict = {}
            for k, v in ckpt['state_dict'].items():
                new_key = k.replace('module.', '', 1)  # 앞에서 한 번만 대체
                new_state_dict[new_key] = v
            tta_model.load_state_dict(new_state_dict, strict=True)
        else:  # non-DDP model
            tta_model.load_state_dict(ckpt['state_dict'])
    
    else: tta_model.load_state_dict(ckpt['model'])

    ### Set criterion
    criterion = nn.CrossEntropyLoss().to(args.device)

    ### TTA
    args.freq = args.input_size // 4
    
    if args.tta_method == 'norm':
        tta_model = norm.Norm(tta_model, alpha = 1.0) # online_tta # no calibration
    elif args.tta_method == 'coda':
        tta_model = coda.configure_model_for_fabn(tta_model, args.freq, device=args.device)
        tta_model = coda.FABN(tta_model)
    elif args.tta_method == 'tent':
        ### Set optimizer & batch size
        tta_model = tent.configure_model_for_tent(tta_model)
        params, _ = tent.collect_params(tta_model)
        optim, kwargs_optim = tent.configure_optimizer_for_tent(args.dataset)
        optimizer = optim(params, **kwargs_optim)
        tta_model = tent.Tent(tta_model, optimizer, steps=1, episodic=False)

    elif args.tta_method == 'sar':  ## hyperparameters not tuned (set for ResNet50, bs > 32)
        tta_model = sar.configure_model(tta_model)
        params, _ = sar.collect_params(tta_model, args.dataset)
        base_optimizer = torch.optim.SGD
        args.lr = 0.00025 ## lr hard-coded
        args.sar_margin_e0 = math.log(1000)*0.40 ## margin_e0 hard-coded
        optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
        tta_model = sar.SAR(tta_model, optimizer, margin_e0=args.sar_margin_e0)

    elif args.tta_method == 'tent_coda':  ## hyperparameters not tuned (set for ResNet50, bs > 32)
        print(f"[INFO] TTA with tent_ + FABN")
        bn_train = args.bn_train           
        tta_model = tent_coda.configure_model(tta_model, args.freq, bn_train, track_running_stats_hfc=True, avg_step=0)
        params, _ = tent_coda.collect_params(tta_model) 
        optim, kwargs_optim = tent.configure_optimizer_for_tent(args.dataset)
        optimizer = optim(params, **kwargs_optim)
        tta_model = tent_coda.TENT_FABN(tta_model, optimizer)

    elif args.tta_method == 'sar_coda':  ## hyperparameters not tuned (set for ResNet50, bs > 32)
        print(f"[INFO] TTA with SAR + FABN")       
        tta_model = sar_coda.configure_model(tta_model, args.freq, avg_step=0)
        params, param_names = sar_coda.collect_params(tta_model, args.dataset)
        base_optimizer = torch.optim.SGD
        args.lr = 0.00025 ## lr hard-coded
        args.sar_margin_e0 = math.log(1000)*0.40 ## margin_e0 hard-coded
        optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
        tta_model = sar_coda.SAR_FABN(tta_model, optimizer, margin_e0=args.sar_margin_e0)

    else:
        raise Exception(f"tta_method {args.tta_method} is not supported.")
    
    tta_model = tta_model.to(args.device)
    
    print("\n[STATUS] Validate after adapt")
    valid(test_loader, tta_model, criterion, 1, args, None, True)  
    
    print("[INFO] Save results")
    save_results(data=args.adapt_result, csv_name='adapt_result.csv', log_path=args.log_path, corr_type=args.corr_type)
        
    ### Save model
    print("[STATUS] Save model")
    torch.save(tta_model.state_dict(), os.path.join(args.save_path, f"{args.corr_type}.pt"))

    
    
if __name__ == "__main__":
    # Set parser
    parser = argparse.ArgumentParser()
    
    # random seed
    parser.add_argument('--project_name', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--yaml', type=str, default='config.yaml', 
                        help='Path to yaml file for additional configurations')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenetc') 
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--shuffle', type=bool, default=True)  # shuffle default True 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--corr_type', type=str, required=True) 
    
    # model
    parser.add_argument('--model_arch', type=str, default='resnet18')
    parser.add_argument('--q_scheme', default='fp')
    parser.add_argument('--q_bits_a', type=int, default=32)
    parser.add_argument('--q_bits_w', type=int, default=32)

    # checkpoint
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the checkpoint file')

    # tta_method
    parser.add_argument('--tta_method', type=str, required=True)
    parser.add_argument('--train_dataset', type=str, default='cifar10') # used in bn_adapter
 
    # result
    
    args = parser.parse_args()
    args = parse_yaml(args) # load yaml as args 

    ### Set device
    args = setup(args)
    
    
    main(args)