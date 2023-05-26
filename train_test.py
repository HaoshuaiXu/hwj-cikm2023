import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from transformers import RobertaTokenizer
from RelationExtractionDataset import RelationExtractionDataset
from RobertaClass import RobertaClass
from train import train


if __name__ == '__main__':
    training_set = pd.read_csv(
        "./dataset/semeval_trainingset_class-code_without-entity.tsv",
        delimiter="\t"
    )
    training_set = RelationExtractionDataset(
        dataframe=training_set,
        tokenizer=RobertaTokenizer.from_pretrained('roberta-base'),
        max_len=256
    )
    training_loader = DataLoader(training_set, batch_size=8)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = RobertaClass()
    model.to(device=device)
    train(
        dataloader=training_loader,
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        device=device,
        test_mode=True # 测试开关
    )