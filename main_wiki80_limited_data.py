import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from RelationExtractionDataset import RelationExtractionDataset
from RobertaClass import RobertaClass
from train import train
from test import test
from caculate_metrics import caculate_output_metrics
from wiki80_preprocess import process_wiki80


if __name__ == "__main__":
    # 读取数据
    training_set = process_wiki80(
        data_filepath="./dataset/wiki80/wiki80_train.txt",
        label2id_filepath="./dataset/wiki80/wiki80_rel2id.json"
    ).sample(frac=0.3, ignore_index=True)
    test_set = process_wiki80(
        data_filepath="./dataset/wiki80/wiki80_val.txt",
        label2id_filepath="./dataset/wiki80/wiki80_rel2id.json"
    )
    
    # 载入 DataLoader
    training_set = RelationExtractionDataset(
        dataframe=training_set,
        tokenizer=RobertaTokenizer.from_pretrained('roberta-base'),
        max_len=256
    )
    test_set = RelationExtractionDataset(
        dataframe=test_set,
        tokenizer=RobertaTokenizer.from_pretrained('roberta-base'),
        max_len=256
    )
    training_loader = DataLoader(training_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True)
    
    # 创建模型
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = RobertaClass(class_num=80)
    model.to(device)
    
    # 训练和测试
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 1e-5
    epochs = 15
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(
            dataloader=training_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            test_mode=False
        )
        y_true, y_pred = test(
            dataloader=test_loader,
            model=model,
            loss_fn=loss_fn,
            device=device,
            test_mode=False
        )
        caculate_output_metrics(y_true, y_pred, epoch)
    print("Done!")