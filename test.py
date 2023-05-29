import torch
from torch.utils.data import DataLoader
import numpy as np


def test(
    dataloader:DataLoader, 
    model:torch.nn.Module, 
    loss_fn, 
    device:str,
    test_mode=False
):
    """返回 [[y_ture], [y_pred]]"""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    true_y_list = []
    pred_y_list = []
    # 在测试集上运行
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            y = data['label'].to(device)
            pred = model(ids, mask, token_type_ids)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            true_y_list.extend(np.array(y.to('cpu')))
            pred_y_list.extend(np.array(pred.argmax(1).to('cpu')))
            if test_mode == False:
                pass
            else:
                break
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return true_y_list, pred_y_list
