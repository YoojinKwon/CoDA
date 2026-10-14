from tqdm import tqdm
import torch.distributed as dist
from modules.utils import *


def valid(valid_loader, model, criterion, epoch, args, index_filter=None, is_adapt=False): 
    # print dataset name
    model.eval()

    losses = AverageMeter("Loss", ":.4e") 
    top1 = AverageMeter("Acc@1", ":6.2f") 
    losses.reset()
    top1.reset()
    progress_bar = tqdm(valid_loader, desc=f"Valid Epoch {epoch}")
    with torch.no_grad():
        if (index_filter is not None) and (args.dataset == 'imagenet_es'):
            for imgs, labels, paths in progress_bar:
                imgs = imgs.to(args.device)
                labels = labels.to(args.device)

                # compute output
                outputs = model(imgs)
                if index_filter is not None: outputs=outputs[:,index_filter]                
                
                loss = criterion(outputs, labels)
                accs = calculate_topk_accs(outputs, labels) # get acc

                # Gather total correct and total loss
                losses.update(loss.item(), imgs.size(0))
                top1.update(accs[0].item(), imgs.size(0))
                
        else:
            for imgs, labels in progress_bar:
                imgs = imgs.to(args.device)
                labels = labels.to(args.device)

                # compute output
                outputs = model(imgs)
                if index_filter is not None: outputs=outputs[:,index_filter]                
                
                loss = criterion(outputs, labels)
                accs = calculate_topk_accs(outputs, labels) # get acc

                # Gather total correct and total loss
                losses.update(loss.item(), imgs.size(0))
                top1.update(accs[0].item(), imgs.size(0))
    if is_adapt:
        args.adapt_result = {'loss': losses.avg, 'acc': top1.avg}
    else:
        args.val_result = {'loss': losses.avg, 'acc': top1.avg}
    print(f"    Loss: {losses.avg:.4e}, Accuracy: {top1.avg:.2f}%")
    # wandb.log({"Validation Loss": losses.avg, "Validation Accuracy": top1.avg})
    return losses.avg, top1.avg


def valid_ddp(valid_loader, model, criterion, epoch, args): 
    model.eval()
    # check_fake_quant_modules(model)

    losses = AverageMeter("Loss", ":.4e") 
    top1 = AverageMeter("Acc@1", ":6.2f") 
    losses.reset()
    top1.reset()

    progress_bar = tqdm(valid_loader, desc=f"Valid Epoch {epoch}", disable=args.local_rank != 0)

    with torch.no_grad():
        for imgs, labels in progress_bar:
            imgs = imgs.cuda(args.local_rank, non_blocking=True)
            labels = labels.cuda(args.local_rank, non_blocking=True)

            # compute output
            outputs = model(imgs)
            
            # print("Outputs (logits):", outputs[0])
            # print("Predicted class:", outputs[0].argmax().item())
            # print("True label:", labels[0].item())
            # break
            loss = criterion(outputs, labels)
            accs = calculate_topk_accs(outputs, labels) # get acc

            # Gather total correct and total loss
            losses.update(loss.item(), imgs.size(0))
            top1.update(accs[0].item(), imgs.size(0))

    dist.barrier()
    print(f"    Loss: {losses.avg:.4e}, Accuracy: {top1.avg:.2f}%")   
    return losses.avg, top1.avg
    
def adapt_and_valid(valid_loader, model, criterion, epoch, args, index_filter=None): 
    model.eval()

    losses = AverageMeter("Loss", ":.4e") 
    top1 = AverageMeter("Acc@1", ":6.2f") 
    losses.reset()
    top1.reset()

    progress_bar = tqdm(valid_loader, desc=f"Valid Epoch {epoch}")
    with torch.no_grad():
        if index_filter:
            for imgs, labels in progress_bar:
                imgs = imgs.to(args.device)
                labels = labels.to(args.device)

                # compute output
                outputs = model(imgs)[:,index_filter]                
                
                loss = criterion(outputs, labels)
                accs = calculate_topk_accs(outputs, labels)
                
                # Gather total correct and total loss
                losses.update(loss.item(), imgs.size(0))
                top1.update(accs[0].item(), imgs.size(0))
        else:
            for imgs, labels in progress_bar:
                imgs = imgs.to(args.device)
                labels = labels.to(args.device)

                # compute output
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                accs = calculate_topk_accs(outputs, labels) # get acc

                # Gather total correct and total loss
                losses.update(loss.item(), imgs.size(0))
                top1.update(accs[0].item(), imgs.size(0))

    args.adapt_result = {'loss': losses.avg, 'acc': top1.avg}
    print(f"    Loss: {losses.avg:.4e}, Accuracy: {top1.avg:.2f}%")
    # wandb.log({"Adapted Validation Loss": losses.avg, "Adapted Validation Accuracy": top1.avg})
