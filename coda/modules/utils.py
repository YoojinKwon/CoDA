import os
import torch
import pandas as pd




def check_fake_quant_modules(model, verbose=True):
    """
    모델 내 FakeQuant 모듈들의 scale/zero_point 상태를 점검합니다.
    주로 validation accuracy가 0일 때 디버깅용으로 사용합니다.
    """
    issues_found = False
    for name, module in model.named_modules():
        if hasattr(module, "scale") and hasattr(module, "zero_point"):
            try:
                scale = module.scale.detach().cpu().numpy()
                zero_point = module.zero_point.detach().cpu().numpy()

                if (scale == 0).any() or (scale < 1e-6).any():
                    issues_found = True
                    print(f"⚠️ [Issue] {name}: scale is zero or too small: {scale}")

                if (zero_point < 0).any() or (zero_point > 255).any():
                    issues_found = True
                    print(f"⚠️ [Issue] {name}: zero_point out of range: {zero_point}")

                if verbose and not issues_found:
                    print(f"✅ {name}: scale={scale}, zero_point={zero_point}")
            except Exception as e:
                print(f"❌ [Error] {name}: Cannot read scale/zero_point - {e}")
    if not issues_found:
        print("✅ All fake quantizers look OK!")

## Helper
def save_results(data, csv_name, log_path, corr_type): # used in test_tta.py
    assert os.path.exists(log_path), "Log path does not exist!"
    csv_path = f"{log_path}/{csv_name}"

    # 새로운 데이터를 DataFrame으로 변환하고 카테고리 열 추가
    data['corr_type'] = corr_type
    data = pd.DataFrame([data])

    # CSV 파일이 이미 존재하는지 확인
    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path)
        updated_data = pd.concat([existing_data, data], ignore_index=True)
    else:
        # 파일이 없으면 새로운 데이터만 저장
        updated_data = data

    # CSV 파일로 저장
    updated_data.to_csv(csv_path, index=False)

    print(f"Results saved to {csv_path}")
    


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def calculate_topk_accs(outputs, targets, topks=(1,)):
    with torch.no_grad():
        max_topk = max(topks)

        _, preds = outputs.topk(max_topk, 1, True, True)
        preds = preds.t()
        corrects = preds.eq(targets.view(1, -1).expand_as(preds))

        accs = []
        for topk in topks:
            acc = corrects[:topk].reshape(-1).float().sum(0, keepdim=False)
            accs.append(acc.mul_(100.0 / targets.size(0)))
        return accs

def correct_indices(outputs, targets, topks=(1,)):
    with torch.no_grad():
        max_topk = max(topks)

        _, preds = outputs.topk(max_topk, 1, True, True)
        preds = preds.t()
        corrects = preds.eq(targets.view(1, -1).expand_as(preds))
        
        return corrects