# train one epoch
from tqdm import tqdm
import torch.distributed as dist
from modules.utils import AverageMeter, calculate_topk_accs
import torch
from modules.frequency import fft3channels_batch

def train(train_loader, model, criterion, optimizer, epoch, args): 
    losses = AverageMeter("Loss", ":.4e") 
    top1 = AverageMeter("Acc@1", ":6.2f") 
    losses.reset()
    top1.reset()

    progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch}")

    # start_time = time.time() # time
    for step, (imgs, labels) in enumerate(progress_bar):
        imgs = imgs.to(args.device)
        labels = labels.to(args.device)

        if args.freq > 0 and args.use_fft:
            batch_size, input_channels, _, _ = imgs.shape
            imgs, _ = fft3channels_batch(imgs, batch_size, input_channels, args.mask, args) # batch, batch_size, input_channels, mask, args
            # mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1) # tiny imagenet
            # std = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)
            # imgs = (imgs - mean) / std
            
        optimizer.zero_grad()        
        outputs = model(imgs)
        
        loss = criterion(outputs, labels)
        accs = calculate_topk_accs(outputs, labels) # get acc
        top1.update(accs[0].item(), imgs.size(0))
        losses.update(loss.item(), imgs.size(0))
        
        loss.backward() # backward
        optimizer.step() # tune parameters
        
    print(f"    Loss: {losses.avg:.4e}, Accuracy: {top1.avg:.2f}%")
    return losses.avg, top1.avg

def train_ddp(train_loader, model, criterion, optimizer, epoch, args): 
    model.train()

    losses = AverageMeter("Loss", ":.4e") 
    top1 = AverageMeter("Acc@1", ":6.2f") 
    losses.reset()
    top1.reset()

    update_freq = args.update_freq  # 기본값 1
    optimizer.zero_grad()

    progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch}", disable=args.local_rank != 0)

    for step, (imgs, labels) in enumerate(progress_bar):
        imgs = imgs.cuda(args.local_rank, non_blocking=True)
        labels = labels.cuda(args.local_rank, non_blocking=True)

        if args.freq > 0 and args.use_fft:
            batch_size, input_channels, _, _ = imgs.shape
            imgs, _ = fft3channels_batch(imgs, batch_size, input_channels, args.mask, args)
            # mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1) # tiny imagenet
            # std = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)
            # imgs = (imgs - mean) / std

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        accs = calculate_topk_accs(outputs, labels)

        loss = loss / update_freq  # loss 평균화
        loss.backward()           # backward만 먼저 누적

        # def check_grad_flow(model):
        #     print("== Gradient Flow Check ==")
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             if param.grad is None:
        #                 print(f"❌ {name}: did NOT receive grad")
        #             elif torch.all(param.grad == 0):
        #                 print(f"⚠️ {name}: grad is ZERO")
        #             else:
        #                 print(f"✅ {name}: grad OK")
        # if args.local_rank == 0:
        #     check_grad_flow(model)  # gradient flow 확인 (디버깅용)
        # stop

        # 누적 후 실제 step 적용
        if (step + 1) % update_freq == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        # metric은 원래의 loss로 기록 (평균된 loss 말고)
        losses.update(loss.item() * update_freq, imgs.size(0))
        top1.update(accs[0].item(), imgs.size(0))

    dist.barrier()
    print(f"    Loss: {losses.avg:.4e}, Accuracy: {top1.avg:.2f}%")

    return losses.avg, top1.avg
    
    
def train_aug(train_loader, model, criterion, optimizer, lr_scheduler, epoch, args): 
    losses = AverageMeter("Loss", ":.4e") 
    top1 = AverageMeter("Acc@1", ":6.2f") 
    losses.reset()
    top1.reset()

    progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch}")

    # start_time = time.time() # time
    for step, (imgs, labels) in enumerate(progress_bar):
        imgs = imgs.to(args.device)
        labels = labels.to(args.device)

        optimizer.zero_grad()        
        outputs = model(imgs)
        
        loss = criterion(outputs, labels)
        accs = calculate_topk_accs(outputs, labels) # get acc
        top1.update(accs[0].item(), imgs.size(0))
        losses.update(loss.item(), imgs.size(0))
        
        loss.backward() # backward
        optimizer.step() # tune parameters
        lr_scheduler.step()
        
    print(f"    Loss: {losses.avg:.4e}, Accuracy: {top1.avg:.2f}%")