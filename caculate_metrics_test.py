import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import pandas as pd
import numpy as np
from RelationExtractionDataset import RelationExtractionDataset
from RobertaClass import RobertaClass
from test import test
from caculate_metrics import caculate_output_metrics


if __name__ == '__main__':
     # 数据
    test_set = pd.read_csv(
        "./dataset/semeval_testset_classcode.tsv",
        delimiter="\t"
    )
    test_set = RelationExtractionDataset(
        dataframe=test_set,
        tokenizer=RobertaTokenizer.from_pretrained('roberta-base'),
        max_len=256
    )
    test_dataloader = DataLoader(test_set, batch_size=4)
    # 模型
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = RobertaClass().to(device)
    # 测试
    loss_fn = torch.nn.CrossEntropyLoss()
    y_true, y_pred = test(
        dataloader=test_dataloader,
        model=model,
        loss_fn=loss_fn,
        test_mode=True,
        device=device
    )
    epoch = 1
    # print(type(np.array(y_true[0].to('cpu'))))
    # print(np.array(y_true[0].to('cpu')))
    caculate_output_metrics(y_true, y_pred, epoch)
    # print(y_pred)
    