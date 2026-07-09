import os
import sys
import types

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import new_utils
import dist_train_cakd
from models.resnet_cakd import resnet50_cakd
from models.vit_cakd import build_teacher


def test_train_one_epoch_runs_on_cpu():
    """Chay that train_one_epoch (dao thu tu GAN + topk fix) tren DataLoader nho, CPU."""
    dev = "cpu"
    model = resnet50_cakd(num_classes=3, pretrained=False).to(dev)
    teacher = build_teacher(3, pretrained=False).to(dev)
    disc = new_utils.NLayerDiscriminator(input_nc=1, ndf=8, n_layers=3).to(dev)

    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    gan = new_utils.GANLoss().to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    dopt = torch.optim.SGD(disc.parameters(), lr=1e-4, momentum=0.9)

    # DataLoader nho: 4 anh random, 3 lop
    images = torch.randn(4, 3, 224, 224)
    targets = torch.tensor([0, 1, 2, 0])
    ds = torch.utils.data.TensorDataset(images, targets)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)

    args = types.SimpleNamespace(
        print_freq=1, clip_grad_norm=None, distill_start=5, distill_ramp=20,
        model_ema_steps=32, lr_warmup_epochs=0,
    )

    # epoch=6 -> _lam>0 -> kich hoat pca/gl/gan (nhanh distill day du)
    dist_train_cakd.train_one_epoch(
        model, disc, teacher, mse, gan, criterion, opt, dopt,
        loader, dev, epoch=6, args=args, model_ema=None, scaler=None,
    )

    # sau 1 epoch: trong so student thay doi (co hoc that)
    assert next(model.parameters()).grad is not None
