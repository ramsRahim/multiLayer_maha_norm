"""ODIN evaluation for ViT-B/16 augreg2 on NINCO.

Key insight: sign(grad w.r.t. CE(logit/T, y)) is the same for all T since
temperature doesn't change argmax. So we compute gradients once per epsilon,
cache the perturbed logits, and sweep T at scoring time.
"""
import torch
import torch.nn as nn
import timm
import numpy as np
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

IMAGENET_VAL = '/home/rahim/exp/Data/imagenet/val'
NINCO_PATH   = '/home/rahim/exp/Data/NINCO/NINCO/NINCO_OOD_classes'
MODEL_NAME   = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
EPSILONS     = [0.0, 0.001, 0.0014, 0.005]
TEMPERATURES = [1, 10, 100, 1000]
BATCH_SIZE   = 64
DEVICE       = 'cuda:0'


def fpr_at_tpr(id_scores, ood_scores, tpr=0.95):
    threshold = np.percentile(id_scores, 100 * (1 - tpr))
    return (ood_scores >= threshold).mean()


def collect_logits(model, loader, epsilons, device, input_std):
    """
    Single pass per epsilon. Returns dict: eps -> logits [N, C].
    For eps=0: no gradient needed, single forward pass.
    For eps>0: forward+backward to get sign grad, then forward on perturbed.
    Since sign(grad) doesn't depend on T, we use T=1 for the gradient step.
    """
    model.eval()
    eps_to_logits = {eps: [] for eps in epsilons}
    nonzero_eps = [e for e in epsilons if e > 0]

    n_batches = len(loader)
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)

        # Compute unperturbed logits (eps=0) and sign gradients
        data.requires_grad_(True)
        logits = model(data)  # [B, C]
        labels = logits.detach().argmax(dim=1)
        loss = nn.CrossEntropyLoss()(logits, labels)  # T=1 for gradient
        loss.backward()

        if 0.0 in eps_to_logits:
            eps_to_logits[0.0].append(logits.detach().cpu())

        if nonzero_eps:
            grad = data.grad.detach()
            sign_grad = torch.ge(grad, 0).float() * 2 - 1  # ±1
            sign_grad[:, 0] /= input_std[0]
            sign_grad[:, 1] /= input_std[1]
            sign_grad[:, 2] /= input_std[2]

            for eps in nonzero_eps:
                perturbed = data.detach() - eps * sign_grad
                with torch.no_grad():
                    logits_p = model(perturbed)
                eps_to_logits[eps].append(logits_p.cpu())

        data.grad = None

        if (batch_idx + 1) % 50 == 0:
            print(f'  batch {batch_idx+1}/{n_batches}', flush=True)

    return {eps: torch.cat(v, dim=0).numpy() for eps, v in eps_to_logits.items()}


def temperature_scores(logits, T):
    """Max softmax confidence with temperature T."""
    scaled = logits / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    softmax = exp / exp.sum(axis=1, keepdims=True)
    return softmax.max(axis=1)


def main():
    print(f'Loading model: {MODEL_NAME}', flush=True)
    model = timm.create_model(MODEL_NAME, pretrained=True).to(DEVICE)
    model.eval()

    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg)
    input_std = list(cfg['std'])
    print(f'Input std: {input_std}', flush=True)

    id_dataset  = dset.ImageFolder(IMAGENET_VAL, transform=transform)
    ood_dataset = dset.ImageFolder(NINCO_PATH,   transform=transform)
    id_loader   = DataLoader(id_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8, pin_memory=True)
    ood_loader  = DataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8, pin_memory=True)
    print(f'ID: {len(id_dataset)} images, OOD: {len(ood_dataset)} images', flush=True)

    print('\nCollecting ID logits...', flush=True)
    id_logits = collect_logits(model, id_loader, EPSILONS, DEVICE, input_std)

    print('\nCollecting OOD logits...', flush=True)
    ood_logits = collect_logits(model, ood_loader, EPSILONS, DEVICE, input_std)

    # Sweep (T, eps)
    print('\n=== ODIN Sweep Results ===', flush=True)
    print(f'{"T":>6}  {"eps":>6}  {"AUROC":>7}  {"FPR@95":>7}', flush=True)
    best_auroc, best_fpr, best_T, best_eps = 0, 100, None, None

    for T in TEMPERATURES:
        for eps in EPSILONS:
            id_s  = temperature_scores(id_logits[eps],  T)
            ood_s = temperature_scores(ood_logits[eps], T)
            labels = np.concatenate([np.ones(len(id_s)), np.zeros(len(ood_s))])
            auroc  = roc_auc_score(labels, np.concatenate([id_s, ood_s])) * 100
            fpr95  = fpr_at_tpr(id_s, ood_s, tpr=0.95) * 100
            print(f'{T:>6}  {eps:>6.4f}  {auroc:>7.2f}  {fpr95:>7.2f}', flush=True)
            if auroc > best_auroc:
                best_auroc, best_fpr, best_T, best_eps = auroc, fpr95, T, eps

    print(f'\nBest: T={best_T}, eps={best_eps}')
    print(f'AUROC : {best_auroc:.2f}%')
    print(f'FPR@95: {best_fpr:.2f}%')


if __name__ == '__main__':
    main()
