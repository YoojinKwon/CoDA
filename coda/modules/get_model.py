# import models
import torch
import models.resnet_lq as resnet_lq
from models.resnet import q_preact_resnet18, q_preact_resnet50, q_preact_resnet26
from models.mobilenet_v3 import qmobilenet_v3_large, qmobilenet_v3_small
from models.efficientnet import qefficientnet_b0

def _lq_ckpt_has_codec_index(ckpt_path):
    # Some LQ checkpoints were saved with the SAT quantizer (has codec_index buffers),
    # others with the non-SAT one; pick the builder whose state dict matches.
    ck = torch.load(ckpt_path, map_location='cpu')
    sd = ck.get('state_dict', ck.get('model', ck))
    return any('codec_index' in k for k in sd)

def get_model(args, num_classes):
    if (args.dataset in ['imagenet', 'imagenetc']) and (args.q_scheme == 'lq'):
        print("Using LQ model for ImageNet")
        use_sat = _lq_ckpt_has_codec_index(args.ckpt_path)
        if args.model_arch == 'resnet18':
            if use_sat:
                model = q_preact_resnet18(args.q_scheme, args.q_bits_w, args.q_bits_a, True, num_classes)
            else:
                model = resnet_lq.q_preact_resnet18(False, args.q_scheme, args.q_bits_w, args.q_bits_a, True, num_classes, args.dataset)
        elif args.model_arch == 'resnet50':
            if use_sat:
                model = q_preact_resnet50(args.q_scheme, args.q_bits_w, args.q_bits_a, True, num_classes)
            else:
                model = resnet_lq.q_preact_resnet50(False, args.q_scheme, args.q_bits_w, args.q_bits_a, True, num_classes, args.dataset)
    else:
        if args.model_arch == 'resnet26':
            model = q_preact_resnet26(args.q_scheme, args.q_bits_w, args.q_bits_a, True, num_classes)
        elif args.model_arch == 'resnet18':
            model = q_preact_resnet18(args.q_scheme, args.q_bits_w, args.q_bits_a, True, num_classes)
        elif args.model_arch == 'resnet50':
            model = q_preact_resnet50(args.q_scheme, args.q_bits_w, args.q_bits_a, True, num_classes)
        elif args.model_arch == 'mobilenetv3_large':
            model = qmobilenet_v3_large(num_classes=num_classes, q_scheme=args.q_scheme, q_bits_w=args.q_bits_w, q_bits_a=args.q_bits_a)
        elif args.model_arch == 'mobilenetv3_small':
            model = qmobilenet_v3_small(num_classes=num_classes, q_scheme=args.q_scheme, q_bits_w=args.q_bits_w, q_bits_a=args.q_bits_a)
        elif args.model_arch == 'efficientnet_b0':
            model = qefficientnet_b0(num_classes=num_classes, q_scheme=args.q_scheme, q_bits_w=args.q_bits_w, q_bits_a=args.q_bits_a)
        else:
            raise Exception(f'Model architecture {args.model_arch} is not supported.')
    return model