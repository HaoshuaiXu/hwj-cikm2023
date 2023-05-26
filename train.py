import torch
from torch.utils.data import dataloader
from torch import nn


def train(
    dataloader:dataloader, 
    model:nn.Module, 
    loss_fn:nn, 
    optimizer:torch.optim,
    device:str,
    test_mode=False,
):
    """
    test_mode: 测试开关，本地电脑跑不动，开了这个测试时就跑一下，后续可以删除
    """
    model.train()
    size = len(dataloader.dataset)
    for batch, data in enumerate(dataloader):
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        y = data['label'].to(device)
        # 计算损失
        pred = model(ids, mask, token_type_ids)
        loss = loss_fn(pred, y)
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # 输出学习情况
        if batch % 100 == 0:
            loss = loss.item()
            current = (batch + 1) * len(y)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")
        if test_mode == False:
            pass
        else:
            break

        