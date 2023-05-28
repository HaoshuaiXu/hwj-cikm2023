import pandas as pd
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import RobertaTokenizer
from RelationExtractionDataset import RelationExtractionDataset
from RobertaClass import RobertaClass
from train import train
from test import test


if __name__ == "__main__":
    # 读取数据
    training_set = pd.read_csv(
        "./dataset/semeval_trainingset_class-code_without-entity.tsv",
        delimeter='\t'
    )
    test_set = pd.read_csv(
        "./dataset/semeval_testset_classcode.tsv",
        delimeter='\t'
    )
    # 载入 DataLoader
    training_set = RelationExtractionDataset(
        dataframe=training_set,
        tokenizer=RobertaTokenizer.from_pretrained('roberta-base'),
        max_len=256
    )
    training_set = RelationExtractionDataset(
        dataframe=test_set,
        tokenizer=RobertaTokenizer.from_pretrained('roberta-base'),
        max_len=256
    )
    training_loader = DataLoader(training_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)
    # 创建模型
    device = 'cuda'
    model = RobertaClass()
    model.to(device)
    # 训练和测试
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 1e-3
    epochs = 5
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
        test(
            dataloader=test_loader,
            model=model,
            loss_fn=loss_fn,
            device=device,
            test_mode=False
        )
    print("Done!")